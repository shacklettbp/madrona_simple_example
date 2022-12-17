#pragma once

namespace madsimple {

enum class CellFlag : uint32_t {
    Nothing = 0,
    Wall    = 1 << 0,
    End     = 1 << 1,
};

struct Cell {
    float reward;
    CellFlag flags;
};

struct GridState {
    const Cell *cells;

    int32_t startX;
    int32_t startY;
    int32_t width;
    int32_t height;
};

inline CellFlag & operator|=(CellFlag &a, CellFlag b)
{
    a = CellFlag(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline bool operator&(CellFlag a, CellFlag b)
{
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) > 0;
}

inline CellFlag operator|(CellFlag a, CellFlag b)
{
    a |= b;

    return a;
}

}
