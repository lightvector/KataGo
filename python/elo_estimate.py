import math
import sys

def win_rate_to_elo(win_rate):
    """Convert a win rate to an Elo rating difference."""
    if win_rate <= 0 or win_rate >= 1:
        raise ValueError("Win rate must be between 0 and 1 exclusively")
    return -400 * math.log10(1/win_rate - 1)

def elo_std_dev(win_rate, n_games):
    """Calculate the standard deviation of the Elo estimate."""
    if n_games <= 0:
        raise ValueError("Number of games must be positive")
    # Standard error of win rate
    win_rate_std = math.sqrt(win_rate * (1 - win_rate) / n_games)
    # Convert to Elo points using the derivative of the Elo function
    elo_derivative = 400 / math.log(10) / (win_rate * (1 - win_rate))
    return abs(elo_derivative * win_rate_std)

def main():
    try:
        win_rate = float(input("输入胜率 (0-1 之间): "))
        n_games = int(input("输入对局数量: "))
        
        elo = win_rate_to_elo(win_rate)
        std_dev = elo_std_dev(win_rate, n_games)
        
        print(f"Elo差值: {elo:.2f} ± {std_dev:.2f}")
    except ValueError as e:
        print(f"错误: {e}")
    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main()