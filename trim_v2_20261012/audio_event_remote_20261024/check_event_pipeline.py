"""Engineering fixtures only: decoding, alignment, masking, training and resume."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import csv
import pickle
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from event_common import *


class DataChecks(unittest.TestCase):
    def test_projection_does_not_fit_validation(self):
        rng = np.random.default_rng(7)
        train,valid = rng.normal(size=(20,9)),rng.normal(size=(8,9))
        a,b,fit = project_features(train,valid)
        a2,b2,fit2 = project_features(train,valid+100)
        np.testing.assert_array_equal(a,a2)
        np.testing.assert_array_equal(fit["mean"],fit2["mean"])
        self.assertFalse(np.array_equal(b,b2))

    def test_derangements_and_path_safety(self):
        for n in (2,3,16,100):
            indices = derangement(n)
            np.testing.assert_array_equal(np.sort(indices),np.arange(n))
            self.assertTrue((indices!=np.arange(n)).all())
        self.assertEqual(relative_media("video$_$1").as_posix(),"video/1.mp4")
        for value in ("../x$_$1","x$_$../1","x","x$_$1$_$2"):
            with self.assertRaises(ValueError):
                relative_media(value)
        with self.assertRaises(ValueError):
            selected({},"test")


class TorchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from runtime_v2 import runtime
        runtime("cpu",2)

    def test_mask_and_protected_routes(self):
        import torch
        from event_model import EventCorrection
        torch.manual_seed(42)
        model = EventCorrection(context_dim=4,event_dim=3).eval()
        n = 6
        base = torch.randn(n,4,2)
        visual_present = torch.tensor([1,1,0,0,1,1],dtype=torch.bool)
        base[~visual_present,3] = base[~visual_present,1]
        data = dict(base_logits=base,text=torch.randn(n,4),audio=torch.randn(n,4),
                    visual=torch.randn(n,4),event=torch.randn(n,3),
                    audio_present=torch.tensor([1,0,1,0,1,1],dtype=torch.bool),
                    visual_present=visual_present)
        torch.testing.assert_close(model(**data),base,atol=0,rtol=0)
        for route in (model.ta,model.tav):
            route["delta"][-1].bias.data.fill_(.5)
        output = model(**data)
        torch.testing.assert_close(output[:,(0,2)],base[:,(0,2)],atol=0,rtol=0)
        torch.testing.assert_close(output[~data["audio_present"]],base[~data["audio_present"]],atol=0,rtol=0)
        torch.testing.assert_close(model(**data,event_present=torch.zeros(n,dtype=torch.bool)),base,atol=0,rtol=0)
        changed = dict(data,visual=torch.randn(n,4)*100)
        torch.testing.assert_close(model(**changed)[:,1],output[:,1],atol=0,rtol=0)
        torch.testing.assert_close(output[~visual_present,3],output[~visual_present,1],atol=0,rtol=0)

    def test_resume_matches_uninterrupted(self):
        import torch
        from event_train import run_adapter
        rng = np.random.default_rng(94)
        n = 20
        cache = dict(ids=np.asarray([f"sample_{i}" for i in range(n)]),labels=np.arange(n)%2,
                     base_logits=rng.normal(size=(n,4,2)).astype(np.float32),
                     text=rng.normal(size=(n,128)).astype(np.float32),
                     audio=rng.normal(size=(n,128)).astype(np.float32),
                     visual=rng.normal(size=(n,128)).astype(np.float32),
                     audio_present=np.ones(n,bool),visual_present=np.ones(n,bool))
        features = rng.normal(size=(n,128)).astype(np.float32)
        cache["base_logits"].fill(0.)
        cache["text"][:,0] = (2*cache["labels"]-1)*10
        features[:,0] = (2*cache["labels"]-1)*10
        with tempfile.TemporaryDirectory(prefix="event_resume_") as temporary:
            root = Path(temporary)
            args = SimpleNamespace(out=root/"full",epochs=3,batch=8,device=torch.device("cpu"),stop_after_epoch=0)
            run_adapter(args,"events",cache,cache,features,features,{"fixture":True})
            args.out = root/"resumed";args.stop_after_epoch = 1
            self.assertEqual(run_adapter(args,"events",cache,cache,features,features,{"fixture":True}),"paused")
            args.stop_after_epoch = 0
            run_adapter(args,"events",cache,cache,features,features,{"fixture":True})
            relative = Path("adapters/events/seed_42")
            a = load_npz(root/"full"/relative/"validation_predictions.npz")
            b = load_npz(root/"resumed"/relative/"validation_predictions.npz")
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])
            full_record = read_json(root/"full"/relative/"complete.json")
            self.assertGreater(full_record["selected_epoch"],0,
                               "Resume comparison must exercise trained weights, not only epoch zero.")
            a_history = list(csv.DictReader((root/"full"/relative/"history.csv").open()))
            b_history = list(csv.DictReader((root/"resumed"/relative/"history.csv").open()))
            self.assertEqual([r["loss"] for r in a_history],[r["loss"] for r in b_history])
            with (args.out/relative/"validation_predictions.npz").open("ab") as handle:
                handle.write(b"modified")
            with self.assertRaisesRegex(RuntimeError,"output changed"):
                run_adapter(args,"events",cache,cache,features,features,{"fixture":True})

    def test_full_fixture(self):
        import torch
        import event_extract
        import event_train
        from event_summarize import summarize
        from transformers import BertConfig,BertModel
        rng = np.random.default_rng(23)
        with tempfile.TemporaryDirectory(prefix="event_fixture_") as temporary:
            root = Path(temporary)
            processed = root/"MSA Datasets/MOSEI/Processed";processed.mkdir(parents=True)
            raw, rows = {}, []
            for split in ("train","valid","test"):
                n = 20
                lab = np.tile([0,2],n//2)
                raw[split] = dict(classification_labels=lab,regression_labels=np.where(lab==0,-.5,.5),
                    id=np.asarray([split+"_video$_$"+str(i) for i in range(n)]),
                    text_bert=np.stack((rng.integers(1,62,size=(n,7)),np.ones((n,7)),np.zeros((n,7))),axis=1).astype(np.int64),
                    audio=rng.normal(size=(n,6,5)).astype(np.float32),
                    vision=rng.normal(size=(n,6,7)).astype(np.float32),
                    audio_lengths=np.full(n,6),vision_lengths=np.full(n,6))
                for i in range(n):
                    rows.append(dict(video_id=split+"_video",clip_id=str(i),mode=split,
                                     label=str(float(raw[split]["regression_labels"][i]))))
            with (processed/"unaligned_50.pkl").open("wb") as handle:
                pickle.dump(raw,handle,protocol=4)
            with (processed.parent/"label.csv").open("w",encoding="utf8",newline="") as handle:
                writer = csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            bert = root/"MSA Datasets/pretrained_berts/bert_en"
            BertModel(BertConfig(vocab_size=64,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
                                intermediate_size=32,max_position_embeddings=32)).save_pretrained(bert)
            raw_root = processed.parent/"Raw"
            waveform = (np.sin(np.arange(32000)*2*np.pi*330/32000)*.1).astype("<f4")
            sample = root/"fixture.mp4"
            subprocess.run([event_extract.ffmpeg_executable(),"-v","error","-f","f32le","-ar","32000",
                            "-ac","1","-i","pipe:0","-c:a","aac","-y",str(sample)],input=waveform.tobytes(),check=True)
            for split in ("train","valid"):
                for i in range(20):
                    destination = raw_root/(split+"_video")/(str(i)+".mp4")
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(sample.read_bytes())
            out = root/"run"
            plan = build_inventory(root,raw_root,out,True)
            self.assertEqual(sum(map(len,plan["splits"].values())),32)
            missing = raw_root/plan["splits"]["train"][0]["relative_path"]
            saved_media = missing.read_bytes();missing.unlink()
            with self.assertRaises(FileNotFoundError):
                build_inventory(root,raw_root,root/"missing",True)
            missing.write_bytes(saved_media)
            assets = root/"assets";assets.mkdir()
            (assets/event_extract.CHECKPOINT_NAME).write_bytes(b"fixture checkpoint")
            (assets/"class_labels_indices.csv").write_bytes((HERE/"assets/class_labels_indices.csv").read_bytes())
            class FakeEncoder(torch.nn.Module):
                def forward(self,wave):
                    value = wave.square().mean(1,keepdim=True)
                    return dict(clipwise_output=value.expand(-1,527),embedding=value.expand(-1,2048))
            arguments = ["event_extract.py","--out",str(out),"--assets",str(assets),"--device","cpu"]
            with patch.object(sys,"argv",arguments),patch.object(event_extract,"load_encoder",return_value=FakeEncoder()):
                event_extract.main()
            with patch.object(sys,"argv",arguments),patch.object(event_extract,"load_encoder",return_value=FakeEncoder()),patch.object(event_extract,"decode_audio",side_effect=AssertionError("Cached clips must not decode again")):
                event_extract.main()
            probe_out = root/"probe"
            probe = build_inventory(root,raw_root,probe_out,False,True,4)
            self.assertTrue(probe["probe"])
            self.assertEqual(set(probe["splits"]),{"train"})
            with patch.object(sys,"argv",["event_extract.py","--out",str(probe_out),
                              "--assets",str(assets),"--device","cpu"]),patch.object(event_extract,"load_encoder",return_value=FakeEncoder()):
                event_extract.main()
            self.assertTrue((probe_out/"event_probe_20261024.zip").is_file())
            self.assertFalse((probe_out/"events_valid.npz").exists())
            with patch.object(sys,"argv",["event_train.py","--root",str(root),
                              "--out",str(probe_out),"--device","cpu"]),self.assertRaisesRegex(RuntimeError,"probe cannot"):
                event_train.main()
            arguments = ["event_train.py","--root",str(root),"--out",str(out),"--device","cpu","--smoke"]
            with patch.object(sys,"argv",arguments):
                event_train.main()
            result = summarize(out)
            self.assertTrue(result["protected_T_TV_exact"])
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(len(result["comparisons"]),4)
            self.assertEqual(len({r["trainable_parameters"] for r in result["selection"]}),1)
            self.assertFalse(any(p.suffix==".pt" for p in (out/"review").rglob("*")))
            modified = load_npz(out/"events_valid.npz")
            modified["ids"] = modified["ids"][::-1]
            atomic_npz(out/"events_valid.npz",**modified)
            with self.assertRaisesRegex(RuntimeError,"output changed"):
                event_train.require_extraction(out)
            self.assertTrue((out/"review/review_manifest.json").is_file())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    dump_json(HERE/"checks/engineering.json",dict(status="passed" if result.wasSuccessful() else "failed",
              test_cases=result.testsRun,failures=len(result.failures),errors=len(result.errors),
              scope="synthetic engineering checks, not benchmark performance"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
