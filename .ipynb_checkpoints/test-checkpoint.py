from tqdm import tqdm
import time

pbar = tqdm(total = 5, position=0, leave=True)


for i in range(5):
    time.sleep(1)
    pbar.update(1)