#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cinttypes>

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

constexpr size_t gigabyte = 1u << 30;
constexpr size_t megabyte = 1u << 20;


struct linear_allocator
{
    u8* memory;
    size_t size;
    size_t offset;

    linear_allocator() = default;

    linear_allocator(size_t total_size) : size(total_size), offset(0)
    {
        memory = static_cast<u8*>(malloc(total_size));
    }

    void* alloc(size_t size_bytes)
    {
        void* ptr = reinterpret_cast<void*>(memory+offset);
        offset += size_bytes;
        assert(offset < size);
        return ptr;
    }

    template <class T>
    T* alloc_n(u32 count)
    {
        return reinterpret_cast<T*>(alloc(sizeof(T) * count));
    }

    void clear()
    {
        offset = 0;
    }

    void print()
    {
        printf("Memory location: %p. Total size: %zu bytes. Current offset: %zu bytes. Next pointer: %p \n", static_cast<void*>(memory), size, offset, static_cast<void*>(memory+offset));
    }

    void* alloc_aligned(size_t size_bytes, size_t alignment)
    {
        size_t mask = alignment - 1;
        assert((alignment & mask) == 0);

        uintptr_t cur_ptr = reinterpret_cast<uintptr_t>(memory+offset);
        uintptr_t misalignment = cur_ptr & mask;
        ptrdiff_t adjustment = misalignment > 0 ? alignment - misalignment : 0;

        size_t size_aligned = (size_bytes + alignment - 1) & ~(alignment - 1);

        void* ptr = reinterpret_cast<void*>(memory+offset+adjustment);
        offset += size_aligned+adjustment;
        assert(offset < size);
        return ptr;
    }
};


using arena = linear_allocator;

void* alloc(arena& arena, size_t size_bytes)
{
    return arena.alloc(size_bytes);
}

template <class T>
T* alloc_n(arena& arena, u32 count)
{
    return arena.alloc_n<T>(count);
}

arena subarena(arena& a, size_t size_bytes)
{
    arena b;
    b.memory = alloc_n<u8>(a, size_bytes);
    b.size   = size_bytes;
    b.offset = 0;
    return b;
}
