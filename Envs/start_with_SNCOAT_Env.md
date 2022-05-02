# Simulation Env
## Scenes
We construct 18 scenes with different types of space non-cooperative object, including asteroids, capsules, rockets, satellites, and stations.
- SNCOAT-Asteroid-(v0-v5)
- SNCOAT-Capsule-(v0-v2) 
- SNCOAT-Rocket-(v0-v2) 
- SNCOAT-Satellite-(v0-v2)
- SNCOAT-Station-(v0-v2)

## SNCOAT-Env-v1

## SNCOAT-Env-v2
Benifitting from PyRep, our SNCOAT-Env-v2 can run in parallel (multi-process) which greatly support the realization of asynchronous deep reinforcement learning.
```
$: cd SNCOAT/Envs
$: python SNCOAT_Env_v2.py
```

## Run SNCOAT-Env-v2 on remote server or cluster
For those asynchronous DRL algorithm, 'PS-Workers' is one of common framework with TensorFlow implementation, in which each worker often need run on one of GPU of remote server or cluster. Of cause, we can set the `num_workers = 1` to run with PC, but in this manner it will greatly slow down the period of overall training, even reach a local maximumn with oscillation. In next, we post how to setup and run `SNCOAT-Env-v2` on remote server (Centos 7 platform).

1. Download CoppeliaSim v4.2 (Linux version) and Setup system env

![图片](https://user-images.githubusercontent.com/20870192/137155478-4c23887c-b0d0-46c6-ad96-a5d0b2a40983.png)
Although, CoppeliaSim only release formal package on Ubuntu platform. We have proven that it can also run on Centos7 platform. Therefore, we suggest you download [CoppeliaSim (Ubuntu 16.04)](https://www.coppeliarobotics.com/downloads#) and copy it to remote server.

And add following command to `~/.bashrc`:
```
export DISPLAY=:0.0
export COPPELIASIM_ROOT=[The absoulute path of your CoppeliaSim directory in remote server]
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

2. Confirm GLIBCXX version

Because CoppeliaSim relies on GLIBCXX 3.4.21, you should first check the GLIBCXX version by following cammand:
```
$: strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX
```
![图片](https://user-images.githubusercontent.com/20870192/137235590-6abd277f-ff7c-4c42-9a6e-8072977419ec.png)

If strings results do not include 3.4.21, we suggest you to compile higher version GCC with source code (GCC 7 or 8). And then, copy new `libstdc++.so.6.0.*` file to `/usr/lib64` and create new soft link named as 'libstdc++.so.6' that direct to `/usr/lib64/libstdc++.so.6.0.*`.

Once you have updated and setupped GCC, you can run previous command to check GLIBCXX version.

![图片](https://user-images.githubusercontent.com/20870192/137235548-289f3a25-8cde-4141-bb27-00322f06a2ad.png)

3. Install VirtualGL
  Download [virtualgl rpm package](https://sourceforge.net/projects/virtualgl/files/2.5.2/) and install it.
  
4. Configure VirtualGL

5. Run CoppeliaSim with Headless mode

```
$: cd CoppeliaSim_PATH
$: ./coppeliaSim -h &
```
If no error is repoted, we can continue to run SNCOAT environment.

6. Run SNCOAT-Env-v2.py

```
$: cd SNCOAT_PATH/Envs
$: python SNCOAT_Env_v2.py
```
The running results have been recorded in './log/log_{thread_num}/SNCOAT_Env_v2/record'

