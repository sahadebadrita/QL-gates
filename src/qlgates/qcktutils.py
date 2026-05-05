import numpy as np
import json
import sys, os
from scipy.linalg import ishermitian
from qiskit import QuantumCircuit, transpile, execute, Aer, IBMQ
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ASAPSchedule
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import transform1, transform2, getRxxgate, getRyygate, getRzzgate

def QSD_transpile1(U):
    # --- Validate unitary ---
    dim = U.shape[0]
    num_qubits = int(np.log2(dim))

    if U.shape[0] != U.shape[1]:
        raise ValueError("Matrix must be square.")

    if 2**num_qubits != dim:
        raise ValueError("Matrix size must be 2^n × 2^n.")

    # check unitarity properly
    if not np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-12):
        raise ValueError("Input matrix is not unitary.")

    # --- Build circuit with unitary ---
    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))

    # --- Transpile to target basis ---
    qc_trans = transpile(
        qc,
        optimization_level=3,
        basis_gates=["rx", "rz", "cx"]
    )

    # Print gate counts
    #print(dict(qc_trans.count_ops()))

    # --- Get synthesized unitary ---
    qc_unitary = np.asarray(execute(qc, Aer.get_backend("unitary_simulator")).result().get_unitary(qc,decimals=16))
    error = np.linalg.norm(U - qc_unitary, 'fro')
    #backend = AerSimulator(method="unitary")
    #result = backend.run(qc_trans).result()
    #qc_unitary = result.get_unitary(qc_trans)

    # --- Remove global phase for fair comparison ---
    phase = np.angle(np.trace(qc_unitary.conj().T @ U))
    qc_unitary *= np.exp(-1j * phase / dim)

    # --- Compute error ---
    error_globalphase = np.linalg.norm(U - qc_unitary, 'fro')

    return qc_trans, error, error_globalphase

def QSD_transpile(U):
    """
    Input: unitary matrix to be decomposed using standard Qiskit func
    Out: transpiled ckt object, transpiled unitary
    """
    # --- Validate unitary ---
    dim = U.shape[0]
    num_qubits = int(np.log2(dim))

    if U.shape[0] != U.shape[1]:
        raise ValueError("Matrix must be square.")

    if 2**num_qubits != dim:
        raise ValueError("Matrix size must be 2^n × 2^n.")

    # check unitarity properly
    if not np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-12):
        raise ValueError("Input matrix is not unitary.")

    # --- Build circuit with unitary ---
    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))

    # --- Transpile to target basis ---
    qc_trans = transpile(
        qc,
        optimization_level=0.0,
        approximation_degree=0.0,
        basis_gates=["rx", "rz", "cx"]
    )
    U_trans = np.asarray(execute(qc_trans, Aer.get_backend("unitary_simulator")).result().get_unitary(qc_trans,decimals=16))

    return qc_trans, U_trans

def schedule_circuit(qc_trans):
    pm = PassManager([ASAPSchedule()])
    qc_sched = pm.run(qc_trans)
    return qc_sched

def qc_to_layer_dicts(qc_trans):
    qc_sched = schedule_circuit(qc_trans)
    dag = circuit_to_dag(qc_sched)
    #dag = circuit_to_dag(qc_trans)
    layer_list = []

    for layer in dag.layers():
        subdag = layer["graph"]
        nodes = list(subdag.op_nodes())

        if not nodes:
            continue

        # ---- CNOT layer ----
        if len(nodes) == 1 and nodes[0].name == "cx":
            node = nodes[0]
            qargs = node.qargs
            layer_list.append({
                "type": "cx",
                "control": qargs[0].index,
                "target": qargs[1].index
            })
            continue

        # ---- Single-qubit layer ----
        L = {"type": "single", "q0": None, "q1": None}

        for node in nodes:
            gate = node.name         # "rx", "rz"
            theta = float(node.op.params[0])
            q = node.qargs[0].index

            L[f"q{q}"] = [gate, theta]

        # fill missing with identity
        if L["q0"] is None: L["q0"] = ["id", 0.0]
        if L["q1"] is None: L["q1"] = ["id", 0.0]

        layer_list.append(L)

    return layer_list

def gates_commute(g0, g1):
    if g0 == "id" or g1 == "id":
        return True
    if g0 == g1:   # Rz with Rz, Rx with Rx
        return True
    return False   # Rx vs Rz do NOT commute


def qc_to_layer_dicts_commuting(qc):
    layers = []
    current = {"q0": None, "q1": None}

    for inst, qargs, _ in qc.data:
        name = inst.name

        # ---------------- CNOT forces a boundary ----------------
        if name == "cx":
            if current["q0"] or current["q1"]:
                layers.append({
                    "type": "single",
                    "q0": current["q0"] or ["id", 0.0],
                    "q1": current["q1"] or ["id", 0.0]
                })
                current = {"q0": None, "q1": None}

            layers.append({
                "type": "cx",
                "control": qargs[0].index,
                "target": qargs[1].index
            })
            continue

        # ---------------- Single-qubit gate ----------------
        q = qargs[0].index
        theta = float(inst.params[0])
        gate = [name, theta]
        key = f"q{q}"
        other_key = "q1" if key == "q0" else "q0"

        # Same qubit already has op → flush layer
        if current[key] is not None:
            layers.append({
                "type": "single",
                "q0": current["q0"] or ["id", 0.0],
                "q1": current["q1"] or ["id", 0.0]
            })
            current = {"q0": None, "q1": None}
            current[key] = gate
            continue

        # Other qubit has op — check commutation
        if current[other_key] is not None:
            g_other = current[other_key][0]
            g_this = gate[0]

            if gates_commute(g_other, g_this):
                current[key] = gate
            else:
                # flush
                layers.append({
                    "type": "single",
                    "q0": current["q0"] or ["id", 0.0],
                    "q1": current["q1"] or ["id", 0.0]
                })
                current = {"q0": None, "q1": None}
                current[key] = gate
        else:
            current[key] = gate

    # flush tail
    if current["q0"] or current["q1"]:
        layers.append({
            "type": "single",
            "q0": current["q0"] or ["id", 0.0],
            "q1": current["q1"] or ["id", 0.0]
        })

    return layers

def save_layers_to_json(layer_list, filename):
    with open(filename, "w") as f:
        json.dump(layer_list, f, indent=4)

def qiskit_to_ql_gate(name):
    if name == "rz":
        return "Rz"
    if name == "rx":
        return "Rx"
    if name == "id":
        return "I"
    raise ValueError(f"Unsupported gate name in QL mapping: {name}")

def build_ql_layer_unitary(layer, n):
    """
    layer: one entry from JSON
    n: nodes per subgraph
    returns: full (2n)^2 x (2n)^2 unitary for this QL layer
    """

    # ----- Two-QL-bit gate (CNOT analogue) -----
    if layer["type"] == "cx":
        # Your QL CNOT = transform2("I","x")
        U_layer = transform2("I", "x", n, theta=None, U=None)
        return U_layer

    # ----- Single-QL-bit layer -----
    gate0_name, theta0 = layer["q0"]
    gate1_name, theta1 = layer["q1"]

    ql0 = qiskit_to_ql_gate(gate0_name)
    ql1 = qiskit_to_ql_gate(gate1_name)

    θ0 = None if ql0 == "I" else theta0
    θ1 = None if ql1 == "I" else theta1

    # Your transform1 returns a (2n)x(2n) matrix
    U0 = transform1(ql0, n, theta=θ0, U=None)
    U1 = transform1(ql1, n, theta=θ1, U=None)

    # Two QL bits → Kronecker product
    U_layer = np.kron(U0, U1)
    return U_layer

def build_ql_circuit_unitary(layers, n):
    dim = (2*n)**2
    U_total = np.eye(dim, dtype=complex)

    for layer in layers:
        U_layer = build_ql_layer_unitary(layer, n)
        U_total = U_layer @ U_total    # natural order: first layer first

    return U_total

def apply_ql_to_R(R, layer_list, n):
    U_total = build_ql_circuit_unitary(layer_list, n)
    if np.allclose(U_total @ U_total.T.conj(),np.eye(U_total.shape[0])):
        print('U is unitary')
    else:
        print('U is not unitary')
    Rg = U_total @ R @ U_total.conj().T
    return U_total,Rg


def expectation(cfg,psi_x,op):
    op1 = transform1(op, cfg.n, theta=None, U=None)
    op2 = transform1('I', cfg.n, theta=None, U=None)
    meas_op = np.kron(op1,op2)
    print(np.shape(psi_x),np.shape(meas_op))
    print(ishermitian(meas_op))
    #meas_op_val = [np.vdot(psi_x[:,i], meas_op @ psi_x[:,i]).real for i in range(2999,psi_x.shape[1],3000)]
    psi_sampled = psi_x[:, ::2999]
    meas_op_val = np.einsum(
        "ij,ji->i",
        psi_sampled.conj().T,
        meas_op @ psi_sampled
    ).real
    print(np.all(np.abs(meas_op_val) <= 1 + 1e-12))
    return meas_op_val


def trotter_rx(n,NQL,h,dt):
    d_site = 2 * n

    # --- Trotter angles ---
    htheta  = 2 * h * (dt / 2.0)  # half-step transverse field

    # --- Single QL-bit gate (Rx) and its tensor power ---
    Rx = transform1('Rx', n, theta=htheta, U=None)    # acts on one QL-bit
    URx = kron_power(Rx, NQL)                         # Rx ⊗ Rx ⊗ ... NQL times
    return URx
