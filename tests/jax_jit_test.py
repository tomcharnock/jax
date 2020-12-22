# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import dtypes
from jax import lib as jaxlib
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import flags
from jax.lib import version
import numpy as np

FLAGS = flags.FLAGS

# These are the scalar elements of `jax.abstract_arrays.array_types`.
# We use strings, because some type can print as another (e.g. np.longlong
# displays as <class 'numpy.int64'>
_SCALAR_NUMPY_TYPES_STR = [
    "bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "complex64", "complex128",
    "longlong", "intc"
]
# We check that it covers all the array types, except for 2.
# If new types are added, this test will fail.
if set([getattr(np, name) for name in _SCALAR_NUMPY_TYPES_STR] +
       [np.ndarray, jax.dtypes.bfloat16]) != jax.abstract_arrays.array_types:
  raise AssertionError("jax_jit_test.py must be exhaustive with respect to "
                       "Jax numpy scalar types. When adding a new supported "
                       "type, you need to update this test file too.")

# Expected values for device_put.
#
# ("--" means "same as for the left entry").
# Input dtype          Output dtype
#                      [jax_enable_x64=True] [jax_enable_x64=False]
# np.bool_             dtype('bool')         --
# np.int8              dtype('int8')         --
# np.int16             dtype('int16')        --
# np.int32             dtype('int32')        --
# np.int64             dtype('int64')        dtype('int32')
# np.uint8             dtype('uint8')        --
# np.uint16            dtype('uint16')       --
# np.uint32            dtype('uint32')       --
# np.uint64            dtype('uint64')       dtype('uint32')
# np.float16           dtype('float16')      --
# np.float32           dtype('float32')      --
# np.float64           dtype('float64')      dtype('float32')
# np.complex64         dtype('complex64')    --
# np.complex128        dtype('complex128')   dtype('complex64')
# np.longlong          dtype('int64')        dtype('int32')
# np.intc              dtype('int32')        --
_EXPECTED_DEVICE_PUT_X64 = {
    value: np.dtype(value) for value in _SCALAR_NUMPY_TYPES_STR
}

_EXPECTED_DEVICE_PUT_NO_X64 = copy.copy(_EXPECTED_DEVICE_PUT_X64)
_EXPECTED_DEVICE_PUT_NO_X64["int64"] = np.int32
_EXPECTED_DEVICE_PUT_NO_X64["uint64"] = np.uint32
_EXPECTED_DEVICE_PUT_NO_X64["float64"] = np.float32
_EXPECTED_DEVICE_PUT_NO_X64["complex128"] = np.complex64
_EXPECTED_DEVICE_PUT_NO_X64["longlong"] = np.int32

_EXPECTED_DEVICE_PUT = {
    True: _EXPECTED_DEVICE_PUT_X64,
    False: _EXPECTED_DEVICE_PUT_NO_X64,
}


def _cpp_device_put(value, device):
  return jaxlib.jax_jit.device_put(value, FLAGS.jax_enable_x64, device)


class JaxJitTest(parameterized.TestCase):

  def test_is_float_0(self):
    if version <= (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")

    self.assertTrue(
        jaxlib.jax_jit._is_float0(np.zeros((5, 5), dtype=jax.float0)))
    self.assertFalse(jaxlib.jax_jit._is_float0(np.zeros((5, 5))))

  def test_DtypeTo32BitDtype(self):
    if version <= (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")
    self.assertEqual(np.float32, jaxlib.jax_jit._DtypeTo32BitDtype(np.float64))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_scalars(self, device_put_function):
    # TODO(jblespiau): Remove after minimal jaxlib version is 0.1.58 or newer.
    if not hasattr(jaxlib.jax_jit, "_device_put"):
      raise unittest.SkipTest("old jaxlib version")

    device = jax.devices()[0]
    for dtype_str in _SCALAR_NUMPY_TYPES_STR:
      dtype = getattr(np, dtype_str)
      # TODO(jblespiau): Add support for these types in the C++ path.
      if dtype == jax.float0 or dtype == jax.dtypes.bfloat16:
        continue
      value = dtype(0)

      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      self.assertEqual(output_buffer.aval, jax.core.ShapedArray((), dtype))
      self.assertEqual(output_buffer.dtype, dtypes.canonicalize_dtype(dtype))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_arrays(self, device_put_function):
    # TODO(jblespiau): Remove after minimal jaxlib version is 0.1.58 or newer.
    if not hasattr(jaxlib.jax_jit, "_device_put"):
      raise unittest.SkipTest("old jaxlib version")

    device = jax.devices()[0]
    for dtype_str in _SCALAR_NUMPY_TYPES_STR:
      dtype = getattr(np, dtype_str)
      # TODO(jblespiau): Add support for these types in the C++ path.
      if dtype == jax.float0 or dtype == jax.dtypes.bfloat16:
        continue
      value = np.zeros((3, 4), dtype=dtype)
      output_bufer = device_put_function(value, device=device)

      self.assertFalse(output_bufer.aval.weak_type)
      self.assertEqual(output_bufer.aval, jax.core.ShapedArray((3, 4), dtype))
      self.assertEqual(output_bufer.dtype, dtypes.canonicalize_dtype(dtype))
      np.testing.assert_array_equal(output_bufer, np.zeros((3, 4), dtype=dtype))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_buffers(self, device_put_function):
    # TODO(jblespiau): Remove after minimal jaxlib version is 0.1.58 or newer.
    if not hasattr(jaxlib.jax_jit, "_device_put"):
      raise unittest.SkipTest("old jaxlib version")

    device = jax.devices()[0]
    jitted_f = jax.jit(lambda x: x + 1)

    # We run it twice, to cover `_DeviceArray` and the C++ `Buffer`.
    for value in range(2):
      buffer = jitted_f(value)
      output_buffer = device_put_function(buffer, device=device)

      self.assertEqual(output_buffer.dtype, buffer.dtype)
      self.assertEqual(output_buffer.aval, buffer.aval)
      np.testing.assert_array_equal(output_buffer, np.array(value + 1))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_sharded_device_array(self, device_put_function):
    device = jax.devices()[0]

    pmaped_f = jax.pmap(lambda x: x + 1)
    for _ in range(2):
      sda = pmaped_f(np.asarray([[1]]))
      output_buffer = device_put_function(sda, device=device)

      self.assertNotIsInstance(output_buffer,
                               jax.interpreters.pxla.ShardedDeviceArray)
      self.assertEqual(output_buffer.dtype, sda.dtype)
      self.assertEqual(output_buffer.aval, sda.aval)
      np.testing.assert_array_equal(output_buffer, np.asarray(sda))

  def test_device_put_on_python_scalars(self):
    # TODO(jblespiau): Remove after minimal jaxlib version is 0.1.58 or newer.
    if not hasattr(jaxlib.jax_jit, "_device_put"):
      raise unittest.SkipTest("old jaxlib version")

    device = jax.devices()[0]
    int_type = dtypes.canonicalize_dtype(np.int64)
    float_type = dtypes.canonicalize_dtype(np.float64)
    complex_type = dtypes.canonicalize_dtype(np.complex128)

    # int
    res = _cpp_device_put(1, device).to_py()
    self.assertEqual(res, 1)
    self.assertEqual(res.dtype, int_type)
    # We also compare to the Python Jax API, to make sure we have the exact
    # same behavior. When Jax removes the flag and removes this feature, this
    # test will fail.
    self.assertEqual(jnp.asarray(1).dtype, res.dtype)

    # float
    res = _cpp_device_put(1.0, device).to_py()
    self.assertEqual(res, 1.0)
    self.assertEqual(res.dtype, float_type)
    self.assertEqual(jnp.asarray(1.0).dtype, res.dtype)

    # bool
    for bool_value in [True, False]:
      res = _cpp_device_put(bool_value, device).to_py()
      self.assertEqual(res, np.asarray(bool_value))
      self.assertEqual(res.dtype, np.bool)
      self.assertEqual(jnp.asarray(bool_value).dtype, res.dtype)

    # Complex
    res = _cpp_device_put(1 + 1j, device).to_py()
    self.assertEqual(res, 1 + 1j)
    self.assertEqual(res.dtype, complex_type)
    self.assertEqual(jnp.asarray(1 + 1j).dtype, res.dtype)

  @unittest.skipIf(jax.lib._xla_extension_version < 2, "jaxlib too old")
  def test_convert_int_overflow(self):
    with self.assertRaisesRegex(OverflowError, "Python int too large.*"):
      jaxlib.jax_jit.device_put(int(1e100), True, jax.devices()[0])

  def test_signature_support(self):
    # TODO(jblespiau): remove after minimal jaxlib version is 0.1.56 or newer.
    if version < (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")

    def f(a, b, c):
      return a + b + c

    jitted_f = jax.api._cpp_jit(f)
    self.assertEqual(inspect.signature(f), inspect.signature(jitted_f))


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
