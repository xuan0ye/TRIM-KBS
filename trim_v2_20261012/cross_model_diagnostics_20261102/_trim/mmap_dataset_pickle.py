"""Read NumPy arrays inside trusted dataset pickle via read-only memory maps.

Large bytes payloads stay on disk. No dataset values are changed.
"""
from pathlib import Path
import pickle,struct
import numpy as np
class Span:
 def __init__(self,path,offset,size):self.path,self.offset,self.size=path,offset,size
class Array:
 def __setstate__(self,state):
  version,shape,dtype,fortran,raw=state
  assert version in (0,1),version
  order='F' if fortran else 'C'
  if isinstance(raw,Span):
   assert not dtype.hasobject
   assert int(np.prod(shape,dtype=np.int64))*dtype.itemsize==raw.size,(shape,dtype,raw.size)
   self.value=np.memmap(raw.path,dtype=dtype,mode='r',offset=raw.offset,shape=shape,order=order)
  elif dtype.hasobject:self.value=np.asarray(raw,dtype=dtype).reshape(shape,order=order)
  else:self.value=np.frombuffer(raw,dtype=dtype).reshape(shape,order=order)
class Reader(pickle._Unpickler):
 dispatch=pickle._Unpickler.dispatch.copy()
 def __init__(self,file):
  super().__init__(file);self.file=file;self.path=Path(file.name)
 def find_class(self,module,name):
  if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':return lambda *args:Array()
  return super().find_class(module,name)
 def payload(self,n):
  if n<1024*1024:return self.read(n)
  frame=self._unframer.current_frame
  assert frame is None or frame.tell()==len(frame.getbuffer()),'Large array unexpectedly embedded in metadata frame'
  offset=self.file.tell();self.file.seek(n,1)
  assert self.file.tell()<=self.path.stat().st_size,'Truncated array payload'
  return Span(self.path,offset,n)
 def load_binbytes(self):self.append(self.payload(struct.unpack('<I',self.read(4))[0]))
 def load_binbytes8(self):self.append(self.payload(struct.unpack('<Q',self.read(8))[0]))
 def load_bytearray8(self):self.append(self.payload(struct.unpack('<Q',self.read(8))[0]))
 dispatch[pickle.BINBYTES[0]]=load_binbytes
 dispatch[pickle.BINBYTES8[0]]=load_binbytes8
 dispatch[pickle.BYTEARRAY8[0]]=load_bytearray8
 def load(self):
  data=super().load()
  def unwrap(x):
   if isinstance(x,Array):return x.value
   if isinstance(x,dict):return {k:unwrap(v) for k,v in x.items()}
   if isinstance(x,list):return [unwrap(v) for v in x]
   if isinstance(x,tuple):return tuple(unwrap(v) for v in x)
   return x
  return unwrap(data)
def load(file):return Reader(file).load()
if __name__=='__main__':
 import sys,time
 start=time.monotonic();path=Path(sys.argv[1])
 with path.open('rb') as f:data=load(f)
 print('Loaded metadata',time.monotonic()-start)
 for split in ('train','valid','test'):
  print(split,{k:(v.shape,str(v.dtype),type(v).__name__) for k,v in data[split].items() if isinstance(v,np.ndarray)})
 print('test labels counts',np.unique(data['test']['regression_labels'],return_counts=True))
