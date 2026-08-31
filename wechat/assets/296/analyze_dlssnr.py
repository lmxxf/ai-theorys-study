#!/usr/bin/env python3
"""Reproduce the local nvngx_dlssnr.dll evidence used by wechat/296.md.

Only the Python standard library is required. The script does not extract or
modify the DLL. It prints its fingerprint, PE sections, resource leaves and a
small set of architecture-relevant ASCII strings.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


INTERESTING_TERMS = (
    b"SwinAttention",
    b"VitAttention",
    b"Downsample",
    b"Upsample",
    b"ConvBlock",
    b"fp8",
    b"sm_120",
    b"WEIGHTS_HT",
)

REPRESENTATIVE_TERMS = (
    b"OBSwinAttention",
    b"CCVitAttention",
    b"CCVit1DAttention",
    b"BSDownsample",
    b"BSUpsampleSkip",
    b"CCDecInputUpsample",
    b"BSFusedConvBlock",
    b"BSGroupedConvBlock",
    b"FusedSubtiledConvBlock",
    b"layer0.conv_weight",
    b"out_conv_weight",
    b"WEIGHTS_HT",
)

WIDE_REPRESENTATIVE_TERMS = (
    "NVSDK_NGX_GPU_Arch_Blackwell2",
    "NVIDIA DLSSNR - DVS PRODUCTION",
    "NVIDIA DLSSNR",
)


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def read_pe(data: bytes) -> tuple[list[dict[str, int | str]], int, int, int]:
    if data[:2] != b"MZ":
        raise ValueError("not a PE file: missing MZ header")
    pe_offset = u32(data, 0x3C)
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("not a PE file: missing PE signature")

    coff = pe_offset + 4
    section_count = u16(data, coff + 2)
    optional_size = u16(data, coff + 16)
    optional = coff + 20
    if u16(data, optional) != 0x20B:
        raise ValueError("expected PE32+ optional header")

    resource_rva = u32(data, optional + 112 + 2 * 8)
    section_table = optional + optional_size
    sections: list[dict[str, int | str]] = []
    for index in range(section_count):
        entry = section_table + index * 40
        name = data[entry : entry + 8].rstrip(b"\0").decode("ascii", "replace")
        sections.append(
            {
                "name": name,
                "virtual_size": u32(data, entry + 8),
                "virtual_address": u32(data, entry + 12),
                "raw_size": u32(data, entry + 16),
                "raw_offset": u32(data, entry + 20),
            }
        )

    for section in sections:
        start = int(section["virtual_address"])
        span = max(int(section["virtual_size"]), int(section["raw_size"]))
        if start <= resource_rva < start + span:
            resource_offset = int(section["raw_offset"]) + resource_rva - start
            return sections, resource_rva, resource_offset, pe_offset
    raise ValueError("resource directory is not mapped by a PE section")


def resource_leaves(
    data: bytes,
    resource_rva: int,
    resource_offset: int,
) -> list[tuple[str, int, int]]:
    leaves: list[tuple[str, int, int]] = []

    def entry_name(value: int) -> str:
        if not value & 0x80000000:
            return str(value)
        offset = resource_offset + (value & 0x7FFFFFFF)
        length = u16(data, offset)
        raw = data[offset + 2 : offset + 2 + length * 2]
        return raw.decode("utf-16le", "replace")

    def walk(relative_offset: int, path: tuple[str, ...] = ()) -> None:
        directory = resource_offset + relative_offset
        count = u16(data, directory + 12) + u16(data, directory + 14)
        for index in range(count):
            entry = directory + 16 + index * 8
            name = entry_name(u32(data, entry))
            target = u32(data, entry + 4)
            if target & 0x80000000:
                walk(target & 0x7FFFFFFF, path + (name,))
                continue
            leaf = resource_offset + target
            data_rva = u32(data, leaf)
            size = u32(data, leaf + 4)
            file_offset = resource_offset + data_rva - resource_rva
            leaves.append(("/".join(path + (name,)), size, file_offset))

    walk(0)
    return leaves


def ascii_strings(data: bytes, minimum: int = 5) -> list[bytes]:
    strings: list[bytes] = []
    start = None
    for index, byte in enumerate(data):
        if 32 <= byte <= 126:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= minimum:
                strings.append(data[start:index])
            start = None
    if start is not None and len(data) - start >= minimum:
        strings.append(data[start:])
    return strings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dll", type=Path)
    args = parser.parse_args()

    data = args.dll.read_bytes()
    sections, resource_rva, resource_offset, _ = read_pe(data)
    leaves = resource_leaves(data, resource_rva, resource_offset)

    print(f"file: {args.dll}")
    print(f"size: {len(data)} bytes")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    print("\nPE sections (virtual/content size, raw size):")
    for section in sections:
        print(
            f"  {section['name']:<8} "
            f"{int(section['virtual_size']):>10} / "
            f"{int(section['raw_size']):>10} bytes"
        )

    print("\nResource leaves:")
    for name, size, offset in leaves:
        print(f"  {name:<24} {size:>10} bytes ({size / 1024 / 1024:.2f} MiB) @ 0x{offset:x}")

    all_strings = ascii_strings(data)
    print("\nArchitecture term counts (matching strings):")
    for term in INTERESTING_TERMS:
        count = sum(term.lower() in string.lower() for string in all_strings)
        print(f"  {term.decode('ascii'):<16} {count}")

    print("\nRepresentative architecture strings:")
    matches = []
    for term in REPRESENTATIVE_TERMS:
        for string in all_strings:
            if term.lower() in string.lower():
                decoded = string.decode("ascii", "replace")
                if decoded not in matches:
                    matches.append(decoded)
                break
    for string in matches:
        print(f"  {string}")

    print("\nRepresentative UTF-16LE strings:")
    for term in WIDE_REPRESENTATIVE_TERMS:
        encoded = term.encode("utf-16le")
        print(f"  {term}: {'yes' if encoded in data else 'no'}")


if __name__ == "__main__":
    main()
