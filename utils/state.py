"""Step state and batch mode, shared by the Slicer steps and run_pipeline.py.
The checkpoint of a step = its output files."""
import os
import sys
import traceback

BATCH_ENV = "SEG2CFD_BATCH"    # "1": launched by run_pipeline.py (Slicer exits when done)
REVIEW_ENV = "SEG2CFD_REVIEW"  # "1": manual review, outputs saved only by save_checkpoint()


def batch():
    return os.environ.get(BATCH_ENV) == "1"


def review():
    return batch() and os.environ.get(REVIEW_ENV) == "1"


def _save_button(step, fn):
    """Toolbar button 'Save and continue' (only with the main window)."""
    import slicer
    mw = slicer.util.mainWindow()
    if mw is None:
        return
    import qt
    tb = mw.addToolBar("seg2cfd")
    act = tb.addAction(f"SAVE {step} AND CONTINUE")
    act.setToolTip("Save the outputs, close Slicer, run_pipeline goes on")
    act.connect("triggered()", fn)
    tb.widgetForAction(act).setStyleSheet("background:#2e7d32; color:white; font-weight:bold; padding:4px")


def finish_step(step, save_fn):
    """Save the step outputs and expose save_checkpoint() in the Slicer console.
    Review mode: nothing saved here; save_checkpoint() (or the toolbar button) saves and closes
    Slicer, and run_pipeline.py goes on. Closing Slicer without saving stops the pipeline."""
    import __main__

    if review():
        import slicer

        def save_and_continue():
            save_fn()
            print(f">>> {step} saved after review, closing Slicer")
            slicer.util.exit(0)

        __main__.__dict__["save_checkpoint"] = save_and_continue
        _save_button(step, save_and_continue)
        print(f"\n>>> {step} done, MANUAL REVIEW: fix the segmentation in the GUI, then press "
              f"'SAVE {step} AND CONTINUE' (or save_checkpoint() in the Python console)")
        return
    save_fn()
    __main__.__dict__["save_checkpoint"] = save_fn
    if batch():
        print(f">>> {step} done, outputs saved")
    else:
        print(f">>> {step} done, outputs saved. After manual corrections: save_checkpoint()")


def reset_3d_view():
    """Centre the 3D view; nothing without a main window (--no-main-window)."""
    import slicer
    lm = slicer.app.layoutManager()
    if lm is not None:
        lm.threeDWidget(0).threeDView().resetFocalPoint()


def run(main, argv, script):
    """Entry point of a Slicer step: main(case_id).
    Interactive: Slicer stays open on the scene (also after an error, to fix it by hand).
    Batch (run_pipeline.py): Slicer exits with 0 on success, 1 on error.
    Review: Slicer stays open (also after an error: fix, then main('<case>') in the console)."""
    if len(argv) != 2:
        raise SystemExit(f"Usage: Slicer --python-script {script} <case_id>")
    if not batch():
        return main(argv[1])
    import faulthandler
    import slicer
    try:  # native crash -> Python stack in the log (Slicer redirects sys.stderr: use the real fd 2)
        faulthandler.enable(file=sys.__stderr__ or open(os.devnull, "w"))
    except Exception:
        pass
    try:
        main(argv[1])
    except Exception:
        traceback.print_exc()
        if review():
            print(f">>> fix it in the GUI, then main('{argv[1]}') in the Python console")
            return
        slicer.util.exit(1)
        return
    if not review():
        slicer.util.exit(0)