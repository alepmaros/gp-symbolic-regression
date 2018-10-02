# Genetic Programming to solve Symbolic Regression

Genetic Programming to solve Symbolic Regression

# How to run:
```
pip install -r requirements.txt
sudo apt install python3-tkinter (If you do not want to use tkinter, remove matplotlib.use('Agg') from all the plots/*)

run.py --help

E.g:
python3 run.py --dataset synth1 -g 100 -p 100 -r 30

Or run all the experiments (Takes a long time)
./experiments.sh
```

By default it runs using 6 cores.

# TO-DO

- [ ] Refactor
- [ ] True parallel implementation
- [ ] Get better results on synth2 and concrete