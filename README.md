# GPMF
Partially Observable Mean Field Multi-Agent Reinforcement  Learning Based on Graph Attention Network for UAV Swarms
This environment contains three confrontation environments: battle, gather, pursuit.

Implementation of GPMF (Graphattention network supported Partially observable Mean Field Multi-agent reinforcement learning) . The paper can be found [here](https://doi.org/10.3390/drones7070476).

The repository is based on [POMFQ](https://github.com/Sriram94/pomfrl).

## Code structure

See folder gamfq for training and testing scripts of the environment.

## Requirements
```linux
Ubuntu 18.04
python==3.6.1
gym==0.9.2
scikit-learn==0.22.0
tensorflow 2
libboost libraries
```
Download the files and store them in a separate directory to build the MAgent framework.
```python
cd /gamfq/examples/battle_model
./build.sh
```

Create a new ```data``` folder and extract ```battle.rar```, ```gather.rar``, ```pursuit.rar``` to the ```gamfq/data```
## Taking the battle environment as an example, gather and pursuit are similar.
### Training 'mfac', 'mfq', 'pomfq', 'gamfq'

```python
cd gamfq  
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
python3 train_battle.py --algo mfq
```

### Testing

Test File: ```battle.py```

```python
runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, battle, pomfq_position, render_every=0)
```

```battle```, ```pomfq_position``` need to be loaded for different models, the specific instructions are as follows:
#### MFQ and MFAC confrontation, that is, mfq/mfac, mfac/mfq
```
battle = battle
python battle.py --algo mfq --oppo mfac --idx 1999 1999 --pomfq_position 2
python battle.py --algo mfac --oppo mfq --idx 1999 1999 --pomfq_position 2
```
#### (MFQ or MFAC) and POMFQ(FOR) confrontation, that is, mfq/pomfq, mfac/pomfq
```
battle = battle
python battle.py --algo mfq --oppo pomfq --idx 1999 1999 --pomfq_position 1
python battle.py --algo mfac --oppo pomfq --idx 1999 1999 --pomfq_position 1
python battle.py --algo pomfq --oppo mfq --idx 1999 1999 --pomfq_position 0
python battle.py --algo pomfq --oppo mfac --idx 1999 1999 --pomfq_position 0
```
#### GPMF and POMFQ(FOR) confrontation, that is, gamfq/pomfq
```
battle = battle2
python battle.py --algo gamfq1 --oppo pomfq --idx 1999 1999 --pomfq_position 2
```
#### POMFQ(FOR) and GPMF confrontation, that is, pomfq/gamfq
```
battle = battle2_change
python battle.py --algo pomfq --oppo gamfq1 --idx 1999 1999 --pomfq_position 2
```
#### GPMF and (MFQ or MFAC) confrontation, that is, gamfq/mfq, gamfq/mfac
```
battle = battle3
python battle.py --algo gamfq1 --oppo mfq --idx 1999 1999 --pomfq_position 2
python battle.py --algo gamfq1 --oppo mfac --idx 1999 1999 --pomfq_position 2
```
#### (MFQ or MFAC) and GPMF confrontation, that is, mfq/gamfq, mfac/gamfq
```
battle = battle3_change
python battle.py --algo mfq --oppo gamfq1 --idx 1999 1999 --pomfq_position 2
python battle.py --algo mfac --oppo gamfq1 --idx 1999 1999 --pomfq_position 2
```

For more help with the installation, look at the instrctions in [MAgent](https://github.com/geek-ai/MAgent), [MFRL](https://github.com/mlii/mfrl) or [POMFQ](https://github.com/Sriram94/pomfrl). In these repsitories installation instructions for OSX is also provided. We have not tested our scripts in OSX.


### Note
This is research code and will not be actively maintained. Please send an email to 2110136@tongji.edu.cn for questions or comments.

## Paper citation
If you found this helpful, please cite the following paper:

```bibtex
@article{yang2023partially,
  title={Partially observable mean field multi-agent reinforcement learning based on graph attention network for UAV swarms},
  author={Yang, Min and Liu, Guanjun and Zhou, Ziyuan and Wang, Jiacun},
  journal={Drones},
  volume={7},
  number={7},
  pages={476},
  year={2023},
  publisher={MDPI}
}
```





