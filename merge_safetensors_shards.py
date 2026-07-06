import argparse
import contextlib
import json
import os
import struct
import sys


def load_safetensors_header(file_path):
    with open(file_path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors file: {file_path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"Incomplete safetensors header: {file_path}")
    return json.loads(header_bytes.decode("utf-8")), 8 + header_length


def copy_range(source_path, start, end, destination_handle, chunk_size=1024 * 1024 * 8):
    remaining = end - start
    with open(source_path, "rb") as source_handle:
        source_handle.seek(start)
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            buffer = source_handle.read(read_size)
            if not buffer:
                raise IOError(f"Unexpected end of file while copying {source_path}")
            destination_handle.write(buffer)
            remaining -= len(buffer)


def build_merged_header(index_path, shard_dir):
    with open(index_path, "r", encoding="utf-8") as handle:
        index_data = json.load(handle)

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Shard index does not contain a valid weight_map")

    merged_header = {}
    metadata = index_data.get("metadata")
    if metadata:
        merged_header["__metadata__"] = metadata

    shard_headers = {}
    data_offset = 0
    copy_plan = []

    for tensor_name, shard_name in weight_map.items():
        shard_path = os.path.join(shard_dir, shard_name)
        if shard_name not in shard_headers:
            header, base_offset = load_safetensors_header(shard_path)
            shard_headers[shard_name] = (header, base_offset)
        header, base_offset = shard_headers[shard_name]
        tensor_info = header.get(tensor_name)
        if tensor_info is None:
            raise KeyError(f"Tensor {tensor_name} was not found in {shard_name}")

        source_start, source_end = tensor_info["data_offsets"]
        tensor_size = source_end - source_start
        merged_header[tensor_name] = {
            "dtype": tensor_info["dtype"],
            "shape": tensor_info["shape"],
            "data_offsets": [data_offset, data_offset + tensor_size],
        }
        copy_plan.append(
            {
                "source_path": shard_path,
                "file_start": base_offset + source_start,
                "file_end": base_offset + source_end,
                "tensor_name": tensor_name,
            }
        )
        data_offset += tensor_size

    header_json = json.dumps(merged_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return header_json, copy_plan


def merge(index_path, shard_dir, output_path):
    header_json, copy_plan = build_merged_header(index_path, shard_dir)
    temp_output = output_path + f".partial.{os.getpid()}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shard_paths = sorted({item["source_path"] for item in copy_plan})
    with contextlib.ExitStack() as stack:
        source_handles = {
            path: stack.enter_context(open(path, "rb"))
            for path in shard_paths
        }
        with open(temp_output, "wb") as handle:
            handle.write(struct.pack("<Q", len(header_json)))
            handle.write(header_json)

            current_shard = None
            for item in copy_plan:
                shard_name = os.path.basename(item["source_path"])
                if shard_name != current_shard:
                    current_shard = shard_name
                    print(f"Copying tensors from {current_shard}")

                source_handle = source_handles[item["source_path"]]
                source_handle.seek(item["file_start"])
                remaining = item["file_end"] - item["file_start"]
                while remaining > 0:
                    read_size = min(1024 * 1024 * 8, remaining)
                    buffer = source_handle.read(read_size)
                    if not buffer:
                        raise IOError(f"Unexpected end of file while copying {item['source_path']}")
                    handle.write(buffer)
                    remaining -= len(buffer)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(temp_output, output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge Hugging Face safetensors shards into one file without loading all tensors into RAM.")
    parser.add_argument("--index", required=True, help="Path to model.safetensors.index.json")
    parser.add_argument("--shard-dir", required=True, help="Directory containing the shard files")
    parser.add_argument("--output", required=True, help="Output safetensors file path")
    args = parser.parse_args()

    try:
        merge(args.index, args.shard_dir, args.output)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Merged safetensors written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())