#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const auto& shape = this->shape();
    const auto& strides = this->strides();
    size_t ndim = shape.size();

    if (ndim == 0) return true; // 标量默认连续

    ptrdiff_t expected_stride = 1;
    // 从最后一维往前推
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<ptrdiff_t>(shape[i]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const auto& old_shape = this->shape();
    const auto& old_strides = this->strides();
    size_t ndim = old_shape.size();

    // 1. 校验 order 长度
    if (order.size() != ndim) {
        throw std::invalid_argument("Permute order size mismatch");
    }

    // 2. 校验 order 合法性（0~ndim-1 无重复）
    std::vector<bool> used(ndim, false);
    for (auto dim : order) {
        if (dim >= ndim || used[dim]) {
            throw std::invalid_argument("Invalid permute order");
        }
        used[dim] = true;
    }

    // 3. 按 order 重排 shape 和 strides
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = old_shape[order[i]];
        new_strides[i] = old_strides[order[i]];
    }

    // 4. 构造新 meta
    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape = new_shape;
    new_meta.strides = new_strides;

    return tensor_t(new Tensor(new_meta, _storage, _offset));
// slice专用：return tensor_t(new Tensor(new_meta, _storage, new_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. 校验元素总数一致
    size_t old_numel = this->numel();
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    if (old_numel != new_numel) {
        throw std::invalid_argument("View shape has different number of elements");
    }

    // 2. 必须是连续的 tensor 才能 view
    if (!this->isContiguous()) {
        throw std::runtime_error("Cannot view a non-contiguous tensor");
    }

    // 3. 生成新的连续 strides
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }

    // 4. 构造新的 meta
    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape = shape;
    new_meta.strides = new_strides;

    // 5. 返回新 tensor，共享 storage 和 offset
   return tensor_t(new Tensor(new_meta, _storage, _offset));
// slice专用：return tensor_t(new Tensor(new_meta, _storage, new_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    const auto& shape = this->shape();
    const auto& strides = this->strides();
    size_t ndim = shape.size();

    // 1. 校验维度合法性
    if (dim >= ndim) {
        throw std::out_of_range("Slice dimension out of range");
    }

    // 2. 校验切片区间合法性
    if (start >= end || end > shape[dim]) {
        throw std::invalid_argument("Invalid slice range");
    }

    // 3. 计算新的 offset（字节偏移）
    ptrdiff_t stride = strides[dim];
    size_t elem_size = this->elementSize();
    // 修正类型转换，避免溢出
    size_t add_offset = static_cast<size_t>(static_cast<int64_t>(start) * stride) * elem_size;
    size_t new_offset = _offset + add_offset;

    // 4. 更新 shape
    std::vector<size_t> new_shape = shape;
    new_shape[dim] = end - start;

    // 5. 构造新 meta，strides 保持不变
    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape = new_shape;
    new_meta.strides = strides;

    // 关键：必须返回 new_offset！
    return tensor_t(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU 设备直接内存拷贝
        std::memcpy(this->data(), src_, this->numel() * this->elementSize());
    } else {
        // 其他设备调用 runtime 的 memcpy
        core::context().runtime().api()->memcpy_sync(
            this->data(),                     // 目标地址：当前 tensor 的数据地址（已含 offset）
            src_,                             // 源地址：CPU 主机数据
            this->numel() * this->elementSize(), // 拷贝总字节数
            LLAISYS_MEMCPY_H2D                // 主机到设备拷贝方向
        );
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
