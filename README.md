# frozen-critic

Can PPO train a randomly initialised actor on GSM8K when its only dense signal is a frozen, pretrained value model?
Not with the frozen 1B critic Limite, which tells the actor only how long to write. With raw values the actor never stops, with
values centred on the prompt it stops at once, and test accuracy is 0 in all six runs. The critic knows the size of the
answer, not the answer.

Limite result: [docs/FINAL.md](docs/FINAL.md). Reopened 2026-10-13 to test larger critics. Notes: [docs/STATUS.md](docs/STATUS.md). Brief:
[docs/BRIEF.md](docs/BRIEF.md). Experiment records: [docs/experiments/](docs/experiments/).
