import faulthandler, sys
faulthandler.enable()
sys.path.insert(0, "/workspace/repo")
# mimic CLI args
sys.argv = ["train_balance.py", "--out", "/tmp/bal_test",
            "--n_envs", "2", "--steps", "40000"]
import train_balance as tb
tb.main()
