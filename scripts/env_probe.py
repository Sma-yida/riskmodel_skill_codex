"""One validation per process; invoked by setup_env.py in the selected Python."""
import argparse
import copy
import importlib.metadata as metadata
import json
import os
import pickle
import platform
import sys
import traceback
from pathlib import Path

MODELING = Path(__file__).resolve().parent / "modeling"
sys.path.insert(0, str(MODELING))
sys.path.insert(0, str(MODELING / "utils"))


def inventory():
    import sysconfig
    import re
    packages = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
                for d in metadata.distributions() if d.metadata["Name"]}
    return {"python": sys.executable, "version": platform.python_version(),
            "python_minor": "%d.%d" % sys.version_info[:2],
            "implementation": platform.python_implementation(), "system": platform.system(),
            "machine": platform.machine().lower(), "platform": sysconfig.get_platform(),
            "packages": dict(sorted(packages.items()))}


def native():
    from lightgbm import LGBMClassifier
    import numpy as np
    from threadpoolctl import threadpool_info
    x = np.arange(100).reshape(50, 2)
    LGBMClassifier(n_estimators=2, n_jobs=2, verbosity=-1).fit(x, [0, 1] * 25)
    return {"openmp_libraries": [p for p in threadpool_info() if p.get("user_api") == "openmp"]}


def grpc_check():
    import grpc
    from concurrent.futures import ThreadPoolExecutor
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    handler = grpc.unary_unary_rpc_method_handler(lambda request, context: request)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("probe", {"echo": handler}),))
    port = server.add_insecure_port("127.0.0.1:0")
    assert port, "gRPC 本机端口创建失败"
    server.start()
    try:
        with grpc.insecure_channel("127.0.0.1:%d" % port) as channel:
            assert channel.unary_unary("/probe/echo")(b"riskmodel", timeout=10) == b"riskmodel"
    finally:
        server.stop(0).wait(5)
    return {"version": grpc.__version__, "local_rpc": True}


def imports():
    import numpy, pandas, sklearn, lightgbm, scorecardpipeline, sklearn2pmml, pypmml, torch, fastai
    from autogluon.tabular import TabularPredictor
    from utils import utils_model_train, utils_model_evaluation, utils_data_quality, utils_data_clear_split, utils_feature_select
    return {p: metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm",
            "autogluon.tabular", "scorecardpipeline", "toad", "sklearn2pmml", "torch", "fastai"]}


def dataset():
    import pandas as pd
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=600, n_features=5, n_informative=4,
                               n_redundant=0, weights=[0.8, 0.2], random_state=17)
    return pd.DataFrame(x, columns=["f1", "f2", "f3", "f4", "f5"]), pd.Series(y, name="label")


def pmml():
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn2pmml import PMMLPipeline, sklearn2pmml
    from sklearn_pandas import DataFrameMapper
    from pypmml import Model
    frame, label = dataset()
    model = PMMLPipeline([("mapper", DataFrameMapper([(c, None) for c in frame.columns])),
                          ("classifier", LGBMClassifier(n_estimators=10, n_jobs=2, verbosity=-1, random_state=17))])
    model.fit(frame.iloc[:450], label.iloc[:450])
    path = Path.cwd() / "probe.pmml"
    sklearn2pmml(model, str(path), with_repr=True)
    assert path.is_file() and path.stat().st_size > 0
    loaded = Model.load(str(path))
    pred = loaded.predict(frame.iloc[450:])
    column = next(c for c in pred.columns if c in ("probability(1)", "probability_1", "probability(1.0)"))
    actual = pred[column].to_numpy(dtype=float)
    expected = model.predict_proba(frame.iloc[450:])[:, 1]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    return {"file_bytes": path.stat().st_size, "prediction_rows": len(pred),
            "max_probability_error": float(np.max(np.abs(actual - expected)))}


def scorecard():
    import numpy as np
    from utils.utils_model_train_scorecard import _fit_scorecard, SCORECARD_DEFAULTS
    frame, label = dataset()
    outputs = {}
    for method in ["chi", "dt", "quantile", "step"]:
        params = copy.deepcopy(SCORECARD_DEFAULTS)
        params["binning"]["method"] = method
        model = _fit_scorecard(frame.iloc[:450], label.iloc[:450], "label", list(frame.columns),
                              [], list(frame.columns), params)
        pred = model.predict_proba(frame.iloc[450:])
        score = model.predict_score(frame.iloc[450:])
        assert np.isfinite(pred).all() and np.isfinite(score).all()
        path = Path.cwd() / ("scorecard-" + method + ".pkl")
        path.write_bytes(pickle.dumps(model))
        loaded = pickle.loads(path.read_bytes())
        np.testing.assert_allclose(pred, loaded.predict_proba(frame.iloc[450:]))
        outputs[method] = {"features": model.feature_names_, "prediction_rows": len(pred)}
    return outputs


def autogluon():
    import numpy as np
    from autogluon.tabular import TabularPredictor
    frame, label = dataset()
    df = frame.copy()
    df["label"] = label
    requested = {"GBM": {"num_boost_round": 5}, "CAT": {"iterations": 5}, "XGB": {"n_estimators": 5},
                 "RF": {"n_estimators": 10}, "XT": {"n_estimators": 10}, "KNN": {},
                 "NN_TORCH": {"num_epochs": 2}, "FASTAI": {"epochs": 2}}
    path = str(Path.cwd() / "autogluon")
    predictor = TabularPredictor(label="label", path=path, eval_metric="roc_auc", verbosity=2)
    predictor.fit(df.iloc[:450], tuning_data=df.iloc[450:], hyperparameters=requested,
                  num_cpus=2, num_gpus=0, fit_weighted_ensemble=False, time_limit=120)
    names = list(predictor.leaderboard(silent=True)["model"])
    expected = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "ExtraTrees", "KNeighbors", "NeuralNetTorch", "NeuralNetFastAI"]
    missing = [n for n in expected if not any(n in model for model in names)]
    assert not missing, {"missing": missing, "trained": names}
    loaded = TabularPredictor.load(path)
    # Verify every requested family, including its persisted prediction path.
    for name in names:
        before = predictor.predict_proba(frame.iloc[:5], model=name)
        after = loaded.predict_proba(frame.iloc[:5], model=name)
        assert np.isfinite(after.to_numpy()).all()
        np.testing.assert_allclose(before, after)
    return {"trained_models": names, "reload_prediction_rows": 5}


CHECKS = {"inventory": inventory, "native": native, "grpc": grpc_check,
          "imports": imports, "pmml": pmml, "scorecard": scorecard, "autogluon": autogluon}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=CHECKS)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    try:
        result = {"ok": True, "details": CHECKS[args.check]()}
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        traceback.print_exc()
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
