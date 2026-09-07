# FatQat Compiler v0.3 设计

> 状态：已按本文方案实现，等待合并审阅  
> 基线：开放仓库 `git@github.com:spaceqat/fatqat.git`，`spaceqat/main@221eb43`  
> 日期：2026-09-07

## 1. v0.3 要解决的问题

v0.2 已经能够把 OpenQASM 送入 SC 或 NA 编译路线，但 Python 用户还不能用一套自然的编程接口直接构造“待编译电路”。现有 `fatqat.Program` 主要承担模拟器和 emulator 的通用执行容器；compiler 内部又有一个同名含义的不可变 `LogicalProgram`。如果要求用户先构造 `fatqat.Program`、再显式转换成 compiler 的 `LogicalProgram`，API 会暴露不必要的内部步骤。

v0.3 的核心目标只有一个：新增面向用户、可编辑的 `fatqat.LogicalProgram`，并让 compiler 直接接受它。用户不需要理解或手工构造任何内部 IR。

```python
import fatqat as fq

program = fq.LogicalProgram(2, 2)
program.h(0)
program.cx(0, 1)
program.measure_all()

compiled = fq.compiler.compile_to_sc(program, backend)
result = backend.run(compiled).result()
```

这项改动增加 compiler frontend，并把已有的 simulator translation 封装进公开 final-target 编译入口；SC、NA、routing 和 native lowering 的分层不变。

## 2. 当前开放仓库的命名基线

此前带有厂商名称的公开接口已经在开放仓库中完成中性化。v0.3 直接沿用这些名称，不再重复改名：

- 公开超导模拟器：`SCQubitSimulator`。名称中的 `Qubit` 是单数。
- 公开超导目标 IR：`SCNativeProgram`，IR ID 为 `sc.native.v1`。
- 公开 QASM 入口：`compile_qasm_to_sc()`。
- 公开 SC pipeline：`qasm-to-sc`。
- 当前公开 native profile 沿用原先的 X/SX/RZ/CZ 基门集合。
- rotation/iSWAP profile 暂时保留为私有实现：`_SCQubitRotationSimulator`、`_RotationNativeProgram`、`_compile_qasm_to_sc_rotation()` 和 `_SC_ROTATION_PIPELINE`。

rotation/iSWAP 路线私有化是产品可见性处理：在 iSWAP 编译一致性问题解决之前，不向普通用户承诺该接口。它不是一次 compiler 分层重构，也不影响统一的 `SCProgram`、路由模型或公开 SC 路线。

后续代码和文档不得重新引入厂商名称作为公开类型或 pipeline 名称。

## 3. 两种 Program 的职责

`fatqat.Program` 和新增的 `fatqat.LogicalProgram` 不应该共用同一个公开类型，因为二者服务于不同边界：

| 类型 | 使用者 | 是否可编辑 | 职责 |
| --- | --- | --- | --- |
| `fatqat.LogicalProgram` | compiler 用户 | 是 | 表达准备编译的、硬件无关的门级电路 |
| `LogicalIR` | compiler passes | 否 | 冻结后的 `gate.logical.v1` 内部 IR |
| `SCProgram` / `NAProgram` | compiler passes | 否 | 已进入具体硬件族语义的中间 IR |
| `SCNativeProgram` / `ZonedPlan` | target lowering | 否 | 目标相关的编译结果 |
| `fatqat.Program` | simulator / emulator | 是 | simulator translation 后可执行的通用容器 |

这里的关键不是复制一套底层表示，而是分开两个公开概念：用户编辑的是 compiler frontend；simulator 执行的是 translation 产物。

## 4. 最小实现结构

### 4.1 用户层采用组合，不采用继承

`LogicalProgram` 内部组合一个现有的 `fatqat.Program`，复用已经成熟的寄存器、operand 解析、门 arity、测量、参数、复制和绘图逻辑：

```python
class LogicalProgram:
    def __init__(self, quantum_registers, classical_registers=0, *, metadata=None):
        self._program = Program(
            quantum_registers,
            classical_registers,
            metadata=metadata,
        )
```

不继承 `Program`，原因是继承会把 simulator/emulator 专用能力和未来设备指令一起暴露成 logical compiler API，也会让下层依赖通过 `isinstance(..., Program)` 意外接受一个尚未 translation 的 frontend 对象。

组合不会改变下层架构。它只是复用 `Program` 的构造能力，并在进入 compiler 时建立清晰边界。`_program` 是实现细节，不属于公共 API。

为了作为显式 pipeline 的输入边界，公开 frontend 声明 `IR_ID = "gate.logical.source.v1"`。这个 ID 标识输入对象类型；冻结后的 dialect 身份仍然是 `gate.logical.v1`，两者不能混用。

### 4.2 内部不可变 IR 改名

当前 `fatqat.compiler.dialects.logical_gate.LogicalProgram` 改名为 `LogicalIR`，其内容和 IR ID `gate.logical.v1` 保持不变：

```python
@dataclass(frozen=True, slots=True)
class LogicalIR:
    IR_ID = "gate.logical.v1"
    qubits: tuple[RegisterRef, ...]
    clbits: tuple[RegisterRef, ...]
    instructions: tuple[LogicalInstruction, ...]
```

这是消除命名冲突，不是新建一个额外编译层。所有现有 logical verifier、SC/NA normalization 只需把类型引用从旧名改为 `LogicalIR`。

### 4.3 自动冻结，不提供手工转换步骤

新增内部 `FreezeLogicalPass`，把可编辑 `LogicalProgram` 快照为 `LogicalIR`。现有 `snapshot_program()` 中已经具备的大部分逻辑应移动或复用到这里，不再让用户调用它。

```text
fatqat.LogicalProgram ── FreezeLogicalPass ──┐
                                             ├─> LogicalIR
OpenQASM ────────────── ParseQasmPass ───────┘
                                                    │
                          ┌─────────────────────────┴────────────────────┐
                          v                                              v
                      SCProgram                                      NAProgram
                          │                                              │
                   routing + native                              ZAP scheduling
                          │                                              │
                          v                                              v
                   SCNativeProgram                                  ZonedPlan
```

QASM 与 Python API 在 `LogicalIR` 汇合，因此不会形成两套 compiler，也不会复制 SC/NA passes。

冻结时按 instruction 顺序生成确定性的 `logical.0`、`logical.1` 等 operation ID。ID 是 compiler provenance，不要求用户提供。对同一份未修改的 `LogicalProgram` 重复编译，必须得到相同 logical IDs；编译过程不得修改源程序。

## 5. 用户 API

### 5.1 构造与寄存器

构造函数与 `fatqat.Program` 保持一致：既接受整数数量，也接受显式 `QuantumRegister` / `ClassicalRegister` 集合。门和测量的 operand 同时接受整数索引或属于本程序的 `RegisterRef`。所有索引、归属、重复 operand 和 arity 检查继续复用现有 `Program` 行为。

```python
program = fq.LogicalProgram(3, 3)
program.h(0)
program.cx(0, 2)
program.measure(0, 0)
```

v0.3 不引入 `LogicalQubit`、`SCQubit` 或另一套寄存器类。logical、SC 和 NA IR 继续使用 `RegisterRef` 保存稳定的程序内引用；进入 target/native 层后再由 layout 和 physical site 表达物理身份。

### 5.2 添加门

方法名采用与 Qiskit Python API 一致的小写形式，避免同时维护大小写两套接口：

- 固定单比特门：`i()`、`h()`、`x()`、`y()`、`z()`、`s()`、`sdg()`、`t()`、`tdg()`、`sx()`。
- 参数门：`rx(theta, qubit)`、`ry(theta, qubit)`、`rz(theta, qubit)`、`phase(theta, qubit)`。
- 双比特门：`cx(control, target)`、`cz(first, second)`、`swap(first, second)`。
- 其它核心指令：`reset(qubit)`、`measure(qubit, clbit)`、`measure_all()`。
- 扩展入口：`add(operation, *qubits)`，供尚未拥有便捷方法的 logical gate 使用。

所有便捷方法都是同一个 `add()` 路径的薄封装，不各自复制检查逻辑。例如：

```python
def h(self, qubit):
    self.add(ops.H, qubit)
    return self

def cx(self, control, target):
    self.add(ops.CX, control, target)
    return self
```

方法返回 `self`，既支持逐行书写，也允许简短链式构造。`add()` 接受 `Operation` 实例；目标是否支持该 operation，由相应 normalization pass 给出明确错误。

v0.3 不把 iSWAP、barrier、pulse、`Put`、`Pair` 或 `Unpair` 加入 logical 便捷 API。前三者尚未形成当前公开 compiler 合同，后三者属于具体设备或 NA 物理调度语义。

### 5.3 当前目标差异

`LogicalProgram` 表达两个公开目标都可能使用的输入集合，但不承诺每个目标都支持每个操作：

- SC normalization 当前支持上述核心集合，包括 `sx` 和 `reset`。
- NA normalization 当前不支持 `sx` 和 `reset`；公开入口以 `PassError` 报告失败，并保留 normalization pass 给出的 unsupported detail。
- 当前 logical IR 只接受有限实数角度。symbolic parameter 在 v0.3 中仍需编译前绑定。
- 测量保持 terminal-only 约束。动态电路、条件执行和中途测量不属于 v0.3。

目标差异由 pass 检查，不在每个 frontend 方法里提前复制一遍 backend 能力判断。

### 5.4 辅助能力

`LogicalProgram` 直接委托以下既有能力：

- `copy()`：返回彼此独立的 `LogicalProgram`。
- `assign_parameters()`：返回完成指定参数绑定的新 `LogicalProgram`，使 symbolic template 能在进入当前 numeric compiler 前完成绑定。
- `draw()`：绘制用户电路。
- `dag()`：按需生成 logical instruction DAG，供观察和算法使用；可编辑电路本身仍以 instruction sequence 为唯一真源。
- `quantum_registers`、`classical_registers` 和 `metadata`：只读暴露寄存器集合，metadata 保持现有可编辑语义。

不公开内部 `_instructions`、`_program` 或手工 `to_logical_ir()` 接口。

## 6. 编译入口

新增两个直接接受 `LogicalProgram` 的公开入口：

```python
fq.compiler.compile_to_sc(
    program: fq.LogicalProgram,
    backend: SCQubitSimulator,
    *,
    emit: str = SCNativeProgram.IR_ID,
    seed: int = 0,
) -> CompilationResult

fq.compiler.compile_to_na(
    program: fq.LogicalProgram,
    architecture: Mapping[str, object],
    *,
    emit: str = ZonedPlan.IR_ID,
) -> CompilationResult
```

默认 final-target 边界实际返回 `ExecutableCompilationResult`。它继承
`CompilationResult`，因此继续提供 `output` 和 `route`，同时携带 translation
生成的 `program` 与 `resource_layout`：

```python
compiled = fq.compiler.compile_to_sc(program, backend)
result = backend.run(compiled).result()

compiled = fq.compiler.compile_to_na(program, architecture)
result = atom_array_backend.run(compiled).result()
```

这一封装同样适用于两个 QASM 入口。显式要求 `LogicalIR`、`SCProgram` 或
`NAProgram` 等中间 `emit` 时，返回值仍是普通 `CompilationResult`，不能直接
执行。`Compiler.compile()` 和两个低层 simulator bridge 保持不变。

模拟器只依赖位于 compiler 包之外的 `ExecutableProgram` 结构契约，从中取出
普通 `fatqat.Program` 与固定 `ResourceLayout`；它不识别
`SCNativeProgram`、`ZonedPlan` 或任何 compiler dialect。因而 bridge 的所有权
仍在 compiler，公共调用只是不再要求用户手工执行 bridge。

`fatqat.__init__` 公开 `LogicalProgram` 和 `compiler` 模块，使 `import fatqat as fq` 后的示例可以直接运行。

已有入口全部保留：

- `compile_qasm_to_sc()`
- `compile_qasm_to_na()`
- `create_sc_pipeline()`
- `create_na_pipeline()`

`compile_to_sc()` 和 `compile_to_na()` 不是先把用户程序序列化成 QASM；它们直接执行 `FreezeLogicalPass`，然后进入与 QASM 路线相同的 `LogicalIR` 之后的 passes。

## 7. Pipeline 注册方式

v0.3 为同一目标注册两个显式 frontend pipeline，避免运行时自动搜索任意类型转换路径：

```text
logical-to-sc: LogicalProgram -> LogicalIR -> SCProgram -> SCNativeProgram
qasm-to-sc:    QasmSource     -> LogicalIR -> SCProgram -> SCNativeProgram

logical-to-na: LogicalProgram -> LogicalIR -> NAProgram -> ZonedPlan
qasm-to-na:    QasmSource     -> LogicalIR -> NAProgram -> ZonedPlan
```

公开便捷函数选择固定 pipeline；不实现双向 type graph、歧义路线搜索或自动猜测输入类型。

## 8. 预计代码改动

实现应保持小范围，主要涉及：

1. 新增 `src/fatqat/logical_program.py`：用户可编辑的 `LogicalProgram` 与门便捷方法。
2. 将 `compiler/dialects/logical_gate.py` 内部类型改名为 `LogicalIR`，IR 内容不变。
3. 将 `compiler/passes/qasm.py` 中已有 snapshot 逻辑抽成共享冻结函数或 pass，使 QASM 和 Python frontend 共用同一条构造路径。
4. 在 `compiler/pipelines.py` 注册 `logical-to-sc`、`logical-to-na`，并新增 `compile_to_sc()`、`compile_to_na()`。
5. 更新 `fatqat/__init__.py`、`fatqat/compiler/__init__.py` 的公开导出。
6. 更新用户 guide 和 API reference；内部设计细节不写入 public API 页面。
7. 更新受 `LogicalProgram` 改名影响的现有 compiler tests。

不新增抽象 factory、gate descriptor registry、backend capability graph、自动 pass 搜索或第二套 operation hierarchy。

## 9. 验收标准

v0.3 完成必须同时满足：

1. 用户只用 `import fatqat as fq` 即可构造 `fq.LogicalProgram` 并调用门方法。
2. 用户程序可直接交给 `compile_to_sc()` 或 `compile_to_na()`，不需要显式 conversion。
3. 等价的 Python `LogicalProgram` 与 OpenQASM 输入冻结后产生等价 `LogicalIR`，除输入来源元数据外不分叉后续 pipeline。
4. operation ID 稳定、唯一；重复编译不改变源程序或前一次结果。
5. integer operand 与 `RegisterRef` operand 的行为一致，非法索引和外部 register ref 在构造时失败。
6. SC 和 NA 的现有成功、拒绝与测量行为不退化。
7. `fatqat.Program` 的 simulator/emulator 公共行为完全不变；默认 final-target
   编译结果可直接传给 matrix simulator 的 `run()`。
8. 公开模块中不出现厂商命名；rotation/iSWAP 路线保持私有。
9. QASM compiler、SABRE、ZAP、simulator bridge 和 visualization 的现有测试继续通过。
10. 新增端到端测试覆盖 Python frontend 到 SC simulator，以及 Python frontend 到 NA simulator 的最小线路。

## 10. v0.3 明确不做的内容

- 不公开 rotation/iSWAP compiler profile。
- 不改变 SABRE、ZAP 或 simulator bridge 的架构。
- 不设计 dynamic circuit、classical feed-forward 或 CFG/region IR。
- 不实现 symbolic parameter 在多层 IR 中的保留与延迟绑定。
- 不增加 pulse-level compiler、noise/calibration-aware routing 或 Duostra。
- 不引入新的 logical/SC/NA qubit 身份类。
- 不让 compiler IR 本身成为 simulator 可执行对象；可执行的是保留 IR 的结果封装。

这些边界保证 v0.3 只改善用户构造和编译入口，不借机扩张 compiler 核心。

## 11. 实施顺序

1. 先将内部 `LogicalProgram` 机械改名为 `LogicalIR`，保持所有现有 compiler 测试通过。
2. 用测试固定 `LogicalProgram` 的构造、门方法、寄存器和复制行为。
3. 实现 `FreezeLogicalPass`，验证确定性 ID、不可变快照和源程序不被修改。
4. 注册 logical-to-SC 与 logical-to-NA pipelines，增加两个公开便捷函数。
5. 做 Python frontend 与 QASM 的等价性测试及两条 simulator 端到端测试。
6. 最后更新公开用户文档；只有通过全部回归后才开放 v0.3 API。
