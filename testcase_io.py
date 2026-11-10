import gzip
import json
import os
import pickle

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector


# Bump this tag when noise generation changes to invalidate stale oracles and caches.
MIXED_STATE_REPRESENTATION = "low_rank_gad_phase_v1"


# Store rho = F F^dagger without allocating a dense density matrix.
class LowRankDensityMatrix:
    def __init__(self, factors):
        factors = np.asarray(factors, dtype=np.complex128)
        if factors.ndim == 1:
            factors = factors[:, np.newaxis]
        if factors.ndim != 2 or factors.shape[1] == 0:
            raise ValueError("factors must have shape (dimension, rank)")

        trace = float(np.real(np.vdot(factors, factors)))
        if not np.isfinite(trace) or trace <= 0:
            raise ValueError("density-matrix factors must have positive finite trace")
        self.factors = factors / np.sqrt(trace)

    @property
    def dim(self):
        return self.factors.shape[0]

    @property
    def rank(self):
        return self.factors.shape[1]

    def purity(self):
        gram = self.factors.conj().T @ self.factors
        return float(np.real(np.vdot(gram, gram)))


def density_factors(state):
    if isinstance(state, LowRankDensityMatrix):
        return state.factors
    data = np.asarray(getattr(state, "data", state))
    if data.ndim == 1:
        return data[:, np.newaxis]
    return None


def low_rank_fidelity(state1, state2):
    factors1 = density_factors(state1)
    factors2 = density_factors(state2)
    if factors1 is None or factors2 is None:
        from qiskit.quantum_info import state_fidelity
        return float(state_fidelity(state1, state2))

    # For a pure-state factor, the only singular value equals the Frobenius norm.
    overlap = factors1.conj().T @ factors2


    if factors1.shape[1] == 1 or factors2.shape[1] == 1:
        fidelity = float(np.real(np.vdot(overlap, overlap)))
        return min(1.0, max(0.0, fidelity))

    singular_values = np.linalg.svd(overlap, compute_uv=False)
    fidelity = float(np.sum(singular_values) ** 2)
    return min(1.0, max(0.0, fidelity))


def low_rank_trace_distance(state1, state2):
    factors1 = density_factors(state1)
    factors2 = density_factors(state2)
    if factors1 is None or factors2 is None:
        rho1 = state1 if isinstance(state1, DensityMatrix) else DensityMatrix(state1)
        rho2 = state2 if isinstance(state2, DensityMatrix) else DensityMatrix(state2)
        return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho1.data - rho2.data))))

    # Evaluate trace distance in the combined factor space, not the full state space.
    combined = np.column_stack((factors1, factors2))
    gram = combined.conj().T @ combined
    gram = (gram + gram.conj().T) / 2
    values, vectors = np.linalg.eigh(gram)
    sqrt_gram = ((vectors * np.sqrt(np.clip(values, 0.0, None))) @
                 vectors.conj().T)
    signs = np.concatenate((
        np.ones(factors1.shape[1]),
        -np.ones(factors2.shape[1]),
    ))
    compressed = (sqrt_gram * signs[np.newaxis, :]) @ sqrt_gram
    compressed = (compressed + compressed.conj().T) / 2
    distance = float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(compressed))))
    return min(1.0, max(0.0, distance))


def _complex_encoder(obj):
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _complex_decoder(dct):
    if "__complex__" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


def resolve_testcase_file(filename):
    """Return an existing testcase file, preferring the requested path."""
    candidates = [filename]

    root, ext = os.path.splitext(filename)
    if ext == ".json":
        candidates.extend([root + ".pkl.gz", root + ".pkl"])
    elif ext == ".gz" and filename.endswith(".pkl.gz"):
        candidates.append(filename[:-7] + ".json")
    elif ext == ".pkl":
        candidates.extend([filename + ".gz", root + ".json"])
    else:
        candidates.extend([filename + ".pkl.gz", filename + ".pkl", filename + ".json"])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"No testcase file found for {filename}")


def _state_to_record(state, precision=np.float32):
    if isinstance(state, LowRankDensityMatrix):
        return {
            "type": "low_rank_density_matrix",
            "data_real": state.factors.real.astype(precision),
            "data_imag": state.factors.imag.astype(precision),
        }
    if isinstance(state, Statevector):
        return {
            "type": "statevector",
            "data_real": state.data.real.astype(precision),
            "data_imag": state.data.imag.astype(precision),
        }
    if isinstance(state, DensityMatrix):
        return {
            "type": "density_matrix",
            "data_real": state.data.real.astype(precision),
            "data_imag": state.data.imag.astype(precision),
        }
    data = np.asarray(state)
    return {
        "type": "ndarray",
        "data_real": data.real.astype(precision),
        "data_imag": data.imag.astype(precision),
    }


def save_test_cases_binary(test_cases, filename, compress=True, precision=np.float32):
    """Save test cases as pickle records with split real/imag arrays."""
    if compress and not filename.endswith(".gz"):
        filename += ".gz"

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    records = []
    for name, state, attributes in test_cases:
        state_record = _state_to_record(state, precision=precision)
        state_record.update({
            "name": name,
            "attributes": attributes,
        })
        records.append(state_record)

    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    return filename


def save_test_cases_json(test_cases, filename):
    """Save test cases in the legacy JSON format."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    data = []
    for name, state, attributes in test_cases:
        if isinstance(state, LowRankDensityMatrix):
            state_data = {
                "type": "low_rank_density_matrix",
                "data": [[_complex_encoder(x) for x in row] for row in state.factors],
            }
        elif isinstance(state, Statevector):
            state_data = {
                "type": "statevector",
                "data": [_complex_encoder(x) for x in state.data],
            }
        elif isinstance(state, DensityMatrix):
            state_data = {
                "type": "density_matrix",
                "data": [[_complex_encoder(x) for x in row] for row in state.data],
            }
        else:
            arr = np.asarray(state)
            state_data = {
                "type": "ndarray",
                "data": [_complex_encoder(x) for x in arr],
            }

        data.append({
            "name": name,
            "state": state_data,
            "attributes": attributes,
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_complex_encoder)

    return filename


def save_test_cases(test_cases, filename, storage_format="auto", compress=True):
    if storage_format == "auto":
        storage_format = "binary" if filename.endswith((".pkl", ".pkl.gz")) else "json"

    if storage_format == "binary":
        return save_test_cases_binary(test_cases, filename, compress=compress)
    if storage_format == "json":
        return save_test_cases_json(test_cases, filename)

    raise ValueError(f"Unsupported storage_format: {storage_format}")


# Recover rank-one states without allocating another dense complex matrix.
def _try_restore_pure_density(data_real, data_imag):
    real = np.asarray(data_real)
    imag = np.asarray(data_imag)
    if real.ndim != 2 or real.shape[0] != real.shape[1]:
        return None

    diagonal = np.diagonal(real)
    trace = float(np.sum(diagonal, dtype=np.float64))
    pivot_index = int(np.argmax(diagonal))
    pivot = float(diagonal[pivot_index])
    if pivot <= 0 or not np.isclose(trace, 1.0, rtol=1e-5, atol=1e-7):
        return None


    column_real = real[:, pivot_index]
    column_imag = imag[:, pivot_index]
    column_norm_sq = column_real * column_real + column_imag * column_imag
    expected_norm_sq = diagonal * pivot
    if not np.allclose(column_norm_sq, expected_norm_sq, rtol=2e-4, atol=1e-10):
        return None

    vector = (column_real.astype(np.complex128) +
              1j * column_imag.astype(np.complex128)) / np.sqrt(pivot)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        return None
    return Statevector(vector / vector_norm)


def _restore_state(state_type, data_real, data_imag, as_numpy=False,
                   compact_pure_density=False):
    if state_type == "density_matrix" and compact_pure_density:
        pure_state = _try_restore_pure_density(data_real, data_imag)
        if pure_state is not None:
            return pure_state

    data = np.asarray(data_real) + 1j * np.asarray(data_imag)
    if state_type == "low_rank_density_matrix":
        return LowRankDensityMatrix(data)
    if as_numpy:
        return data
    if state_type == "density_matrix":
        return DensityMatrix(data)
    return Statevector(data)


def _testcase_type(name, attributes):
    name_parts = name.split('_', 2)
    case_type = name_parts[0]
    if case_type == "supplement" and len(name_parts) > 1:
        case_type = name_parts[1]
    if case_type not in {"classical", "superposition", "entangled", "mixed"}:
        case_type = attributes.get("type", "classical")
    return case_type


def _load_json_records(filename, as_numpy=False, exclude_types=None):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f, object_hook=_complex_decoder)

    test_cases = []
    for entry in data:
        if _testcase_type(entry["name"], entry["attributes"]) in exclude_types:
            continue
        state_data = entry["state"]
        state_type = state_data["type"]
        data_array = np.array(state_data["data"], dtype=complex)
        if state_type == "low_rank_density_matrix":
            state = LowRankDensityMatrix(data_array)
        elif as_numpy:
            state = data_array
        elif state_type == "density_matrix":
            state = DensityMatrix(data_array)
        else:
            state = Statevector(data_array)

        test_cases.append((entry["name"], state, entry["attributes"]))

    return test_cases


def _load_binary_records(filename, as_numpy=False, exclude_types=None,
                         compact_pure_density=False):
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rb") as f:
        records = pickle.load(f)

    test_cases = []
    for record in records:


        if _testcase_type(record["name"], record["attributes"]) in exclude_types:
            continue
        state = _restore_state(
            record["type"],
            record["data_real"],
            record["data_imag"],
            as_numpy=as_numpy,
            compact_pure_density=compact_pure_density,
        )
        test_cases.append((record["name"], state, record["attributes"]))

    return test_cases


def load_test_cases(filename, as_dict=False, as_numpy=False, exclude_types=None,
                    compact_pure_density=False):

    exclude_types = set(exclude_types or ())
    resolved = resolve_testcase_file(filename)
    if resolved.endswith(".json"):
        test_cases = _load_json_records(
            resolved, as_numpy=as_numpy, exclude_types=exclude_types
        )
    elif resolved.endswith((".pkl", ".pkl.gz")):
        test_cases = _load_binary_records(
            resolved,
            as_numpy=as_numpy,
            exclude_types=exclude_types,
            compact_pure_density=compact_pure_density,
        )
    else:
        raise ValueError(f"Unsupported testcase file type: {resolved}")

    if as_dict:
        return {
            name: {
                "state": state,
                "attributes": attributes,
            }
            for name, state, attributes in test_cases
        }

    return test_cases
