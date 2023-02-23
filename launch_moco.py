import os
import sys
import multiprocessing as mp


def f(rank_id, args):
    os.environ["RANK_ID"] = f"{rank_id}"  # logical id
    os.environ["DEVICE_ID"] = f"{rank_id}"  # physical id
    print(f"Launching rank: {os.getenv('RANK_ID')}, device: {os.getenv('DEVICE_ID')}, pid: {os.getpid()}")
    os.system(f"python -u main_moco.py {args}")


if __name__ == '__main__':
    mp.set_start_method("spawn")

    RANK_SIZE = 8
    os.environ["RANK_TABLE_FILE"] = "/path/to/rank_table.json"
    print(f"Args: {' '.join(sys.argv[1:])}")

    processes = [mp.Process(target=f, args=(i, ' '.join(sys.argv[1:]))) for i in range(RANK_SIZE)]
    [p.start() for p in processes]
    [p.join() for p in processes]

# 8P
# python launch_moco.py --data /path/to/IN1K --output-dir ./output/moco
# 1P
# python main_moco.py --data /path/to/IN1K --output-dir ./output/moco --distributed=False
