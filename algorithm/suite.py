"""Execute the predefined required experiments and optional controlled studies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from .main import choose_device, positive_int, DATA_DIR
    from .artifacts import atomic_json, source_identity, identity
except ImportError:
    from main import choose_device, positive_int, DATA_DIR
    from artifacts import atomic_json, source_identity, identity

REQUIRED = [("fashion0.3", False, "trimmed"), ("fashion0.6", False, "trimmed"), ("cifar", True, "trimmed")]
EXTENDED = REQUIRED + [("fashion0.3", True, "trimmed"), ("fashion0.6", True, "trimmed")] + [
    (dataset, True, "argmax") for dataset in ("fashion0.3", "fashion0.6", "cifar")]


def main():
    """Run the required or extended experiment list, one subprocess per experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("required", "extended"), default="extended")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--revision-epochs", type=positive_int)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-limit", type=positive_int)
    parser.add_argument("--test-limit", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.smoke and (args.runs < 10 or args.sample_limit or args.test_limit):
        parser.error("Final suite requires at least ten full-data runs; otherwise use --smoke")
    device = choose_device(args.device)
    jobs = EXTENDED if args.suite == "extended" else REQUIRED
    definition = {"source": source_identity(), "suite": args.suite, "device": device,
                  "runs": args.runs, "epochs": args.epochs, "revision_epochs": args.revision_epochs,
                  "threads": args.threads, "smoke": args.smoke, "sample_limit": args.sample_limit,
                  "test_limit": args.test_limit}
    signature = identity(definition)
    output = args.output.resolve()
    manifest = output / "suite.json"
    if output.exists():
        if not args.resume:
            parser.error("Output exists; choose a new output or explicitly use --resume")
        if not manifest.is_file() or json.loads(manifest.read_text())["signature"] != signature:
            parser.error("Suite definition changed; choose a new output")
    else:
        output.mkdir(parents=True)
        atomic_json(manifest, {"signature": signature, "definition": definition, "jobs": jobs})
    for dataset, estimated, estimator in jobs:
        name = f"{dataset}_{'estimated' if estimated else 'given'}_{estimator}"
        destination = output / name
        command = [sys.executable, "-u", str(Path(__file__).with_name("main.py")),
                   "--dataset", dataset, "--data-dir", str(args.data_dir.resolve()), "--output", str(destination),
                   "--runs", str(args.runs), "--threads", str(args.threads), "--device", device,
                   "--estimator", estimator]
        if estimated:
            command.append("--estimated")
        if args.smoke:
            command.append("--smoke")
        if args.resume and destination.exists():
            command.append("--resume")
        for option in ("epochs", "revision_epochs", "sample_limit", "test_limit"):
            value = getattr(args, option)
            if value is not None:
                command.extend(["--" + option.replace("_", "-"), str(value)])
        print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
        with (output / (name + ".log")).open("a" if args.resume else "x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            parser.exit(result.returncode, f"Failed: {name}; see its log. Completed experiments are preserved.\n")
        print(f"Completed {name}", flush=True)
    print("Suite complete. Run analyse.py and verify.py against this output.", flush=True)


if __name__ == "__main__":
    main()
