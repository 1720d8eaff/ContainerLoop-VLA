# G1-D 容器操作：MuJoCo 仿真

本目录保存一次新的仿真研究运行，使用仓库提供的 G1-D URDF 关节结构与惯性、固定底座和重力补偿；夹爪、烧杯和锥形烧瓶为碰撞模型。；观测使用仿真真实状态。

## 任务与方法

机器人左臂通过 IK 与关节位置控制，夹持左侧盛有小球的烧杯，搬运并倾倒至右侧锥形烧瓶。48 个条件分别覆盖四种杯体质量比例、三种夹持摩擦和四种初始位置；每个条件比较 direct、smoothing、geometry、feedback、full 五种方法。各组使用相同初始条件及随机种子。

成功判定：小球中心从上方穿越瓶口平面，穿越点满足考虑球半径的径向净空；小球最终位于瓶内且速度小于 0.08 m/s。每回合保留成功或失败、条件、种子、瓶口穿越、最终位置、速度、抓取误差、CEM 候选及轨迹文件名。抓取误差是稳定双指接触时夹爪中心与估计烧杯抓取中心的三维距离，不是搬运中的累计滑移。

## 本次运行

| 方法 | 48 组成功数 | 成功率 |
| --- | ---: | ---: |
| direct | 12/48 | 25.0% |
| smoothing | 14/48 | 29.2% |
| geometry | 13/48 | 27.1% |
| feedback | 12/48 | 25.0% |
| full | 41/48 | 85.4% |

`main_run/` 是本次重新运行的 240 回合。`validation/` 是此前完成的 12 个新条件、每组五种方法的 60 回合，full 为 9/12；两组使用相同仿真模型，规模不足以代表真实机器人泛化性能。主实验曾用于探索方案，不是盲测。保留了全部成功和失败回合；轨迹在每 2 ms 输出一次，落出工作台后没有地板，因此可能出现负高度。

## 文件与复现

- `benchmark.py`：主实验场景控制、CEM 规划与记录。
- `benchmark_holdout.py`：额外 12 条条件的验证入口。
- `g1d_pour_prototype.xml`：简化 G1-D 与物理场景模型。
- `optim.py`：CEM 实现。
- `previous_cartesian_contact_benchmark.py`：此前的笛卡尔夹爪代理版本，保留为独立参考，不能产生上述 G1-D 结果。
- `main_run/episodes.jsonl`、`main_run/summary.json`、`main_run/summary.jsonl`：完整逐回合记录及同一汇总的 JSON / 单行 JSONL 形式。
- `main_run/traces/*.jsonl.gz`：240 个逐步轨迹；`validation/` 提供相同结构的 60 个验证轨迹。
- `trace_manifest.json`：每个压缩和解压后轨迹的 SHA-256、文件大小和步数。

轨迹在 Git 中逐回合 gzip 存储，`episodes.jsonl` 中的 `trace_file` 是解压后相对于所在运行目录的路径；在对应 `main_run/` 或 `validation/` 目录执行以下命令可解压原始 JSONL，已有同名文件不会覆写：

```bash
python -c "import gzip,pathlib,shutil; files=pathlib.Path('traces').glob('*.jsonl.gz'); [(lambda src,dst: (shutil.copyfileobj(src,dst),src.close(),dst.close()))(gzip.open(p,'rb'),p.with_suffix('').open('xb')) for p in files]"
```

建议使用 Python 3.12、MuJoCo 3.17.0、NumPy 1.26.4 和 SciPy。在本目录安装依赖后运行：

```bash
python -m pip install -r ../../requirements.txt
python benchmark.py --output new_main --workers 6
python benchmark_holdout.py --output new_validation --workers 6
```

输出目录必须不存在。记录中的 `model_sha256`、`script_sha256` 标识当次模型和脚本；不同依赖版本与计算环境可能影响数值。逐步日志包括小球速度、烧杯姿态与相对夹爪位置，但没有全部关节状态、力矩或每个碰撞对的身份。
