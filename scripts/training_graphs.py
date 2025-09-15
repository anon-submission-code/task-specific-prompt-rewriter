import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(infile: str, window: int = 10, save_path: str = None):
    # 1) Load your log file
    df = pd.read_csv(
        infile,
        sep=r'\s+',            # whitespace delimiter
        header=None,
        names=['timestamp', 'reward', 'step'],
        dtype={'timestamp': float, 'reward': float, 'step': int}
    ).sort_values('step')

    # 2) Compute running mean
    df['running_mean'] = df['reward'].rolling(window=window, min_periods=1).mean()

    # 3) Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['reward'], label='Raw reward')
    plt.plot(df['step'], df['running_mean'], label=f'{window}-step running mean')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward vs. Step')
    plt.legend(loc='best')
    plt.tight_layout()

    # 4) Show or save
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add a n-step running mean of the reward column")
    parser.add_argument('--infile', default="rewards.csv", help="Input file (whitespace-delimited)")
    parser.add_argument('--window', type=int, default=10,
                        help="Rolling window size (default: 10)")
    args = parser.parse_args()
    main(args.infile, args.window)
