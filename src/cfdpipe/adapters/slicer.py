# seg-to-cfd/adapters/slicer.py
import subprocess
from pathlib import Path
from .base import Adapter
import subprocess
from typing import Optional

class SlicerInteractiveAdapter(Adapter):
    def __init__(self, stage: str, slicer_bin: str, script: Path, extensions_length: float):
        self.stage = stage
        self.slicer_bin = slicer_bin
        self.script = script
        self.extensions_length = extensions_length

    def preconditions(self, patient) -> None:
        input_file = patient.path.resolve() / "lumen_tree_cfd.vtk"
        if not input_file.exists():
            raise FileNotFoundError(f"Input non trovato: {input_file}")
        if not Path(self.slicer_bin).exists():
            raise FileNotFoundError(
                f"Slicer non trovato: {self.slicer_bin} (vedi config/paths.yaml)"
            )
        if not self.script.exists():
            raise FileNotFoundError(f"Script Slicer non trovato: {self.script}")

    def run(self, patient) -> None:
        patient_dir = str(patient.path.resolve())
        cmd = [
            self.slicer_bin,
            "--no-splash",
            "--python-script", str(self.script),
            "--",
            "--patient-dir", patient_dir,
            "--flow-ext", str(self.extensions_length)
        ]
        print(f"[DEBUG] SlicerInteractiveAdapter.run: cmd={' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            print(f"[DEBUG][Slicer] {line.rstrip()}")
        process.wait()
        output = "".join(output_lines)
        print(f"[DEBUG] SlicerInteractiveAdapter.run: returncode={process.returncode}")

        # Dump full Slicer stdout/stderr to a patient-local log for post-mortem
        try:
            from pathlib import Path
            log_path = Path(patient_dir) / "slicer.log"
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(output)
            print(f"[DEBUG] Wrote slicer log to: {log_path}")
        except Exception as e:
            print(f"[DEBUG] Failed to write slicer log: {e}")
        if process.returncode != 0:
            raise RuntimeError(
                f"Errore durante l'esecuzione di Slicer su {patient.id}: codice {process.returncode}\n{output}"
            )
        # Se Slicer torna 0, non consideriamo fatali i messaggi di mancata
        # istanziazione VMTK (es. 'Fail to instantiate module'). Questi sono
        # spesso warning legati a estensioni optional; consideriamo fatali solo
        # errori d'import pesanti che impediscono l'esecuzione di Python.
        fatal_tokens = ["ImportError:", "cannot import name"]
        if any(token in output for token in fatal_tokens):
            raise RuntimeError(
                f"Errore durante l'esecuzione di Slicer su {patient.id}: rilevato errore nel log\n{output}"
            )

    def validate(self, patient) -> dict[str, str]:
        endpoints = patient.path / f"Endpoints_{patient.id}.mrk.json"
        tree_model = patient.path / f"tree_model_{patient.id}.vtk"
        cap_model = patient.path / "lumen_tree_cfd_cap.vtk"

        status_report = {
            "endpoints_exists": endpoints.exists(),
            "tree_model_exists": tree_model.exists(),
            "cap_model_exists": cap_model.exists(),
        }
        print(f"[DEBUG] SlicerInteractiveAdapter.validate: {status_report}")

        artifacts: dict[str, str] = {}
        if endpoints.exists():
            artifacts["endpoints"] = str(endpoints)
        if tree_model.exists():
            artifacts["tree_model"] = str(tree_model)
        if cap_model.exists():
            artifacts["cap_model"] = str(cap_model)

        if not artifacts:
            raise FileNotFoundError(f"Artifacts mancanti per {patient.id}: {status_report}")

        return artifacts


class SlicerConversionAdapter(Adapter):
    """Adapter minimale per eseguire lo script Slicer che converte
    `Combined.seg.nrrd` -> `lumen_tree_cfd.vtk`.
    """
    def __init__(self, stage: str, slicer_bin: str, script: Path, output_filename: str = "lumen_tree_cfd.vtk"):
        self.stage = stage
        self.slicer_bin = slicer_bin
        self.script = script
        self.output_filename = output_filename

    def preconditions(self, patient) -> None:
        # Verify slicer binary and script exist and input segmentation is present
        input_file = Path(patient.path.resolve()) / "Combined.seg.nrrd"
        if not input_file.exists():
            raise FileNotFoundError(f"Input segmentation non trovato: {input_file}")
        if not Path(self.slicer_bin).exists():
            raise FileNotFoundError(f"Slicer non trovato: {self.slicer_bin} (vedi config/paths.yaml)")
        if not self.script.exists():
            raise FileNotFoundError(f"Script Slicer non trovato: {self.script}")

    def run(self, patient) -> None:
        patient_dir = str(patient.path.resolve())
        # Pass patient dir as last argument (export_lumen.py legge sys.argv[-1])
        cmd = [
            self.slicer_bin,
            "--no-splash",
            "--python-script",
            str(self.script),
            "--",
            patient_dir,
        ]
        print(f"[DEBUG] SlicerConversionAdapter.run: cmd={' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            print(f"[DEBUG][Slicer-conv] {line.rstrip()}")
        process.wait()
        output = "".join(output_lines)
        print(f"[DEBUG] SlicerConversionAdapter.run: returncode={process.returncode}")

        try:
            log_path = Path(patient_dir) / "slicer_conversion.log"
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(output)
            print(f"[DEBUG] Wrote slicer conversion log to: {log_path}")
        except Exception as e:
            print(f"[DEBUG] Failed to write slicer conversion log: {e}")

        if process.returncode != 0:
            raise RuntimeError(
                f"Errore durante l'esecuzione di Slicer conversion su {patient.id}: codice {process.returncode}\n{output}"
            )

    def validate(self, patient) -> dict[str, str]:
        out = Path(patient.path) / self.output_filename
        exists = out.exists()
        status_report = {"conversion_exists": exists}
        print(f"[DEBUG] SlicerConversionAdapter.validate: {status_report}")
        artifacts: dict[str, str] = {}
        if exists:
            artifacts["lumen_tree"] = str(out)
            return artifacts
        raise FileNotFoundError(f"Artifact di conversione mancante: {out}")