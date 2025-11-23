import numpy as np
import os

DIRECTIONS = np.array([
    [2,0],
    [1,1],
    [0,2],
    [-1,1],
    [-2,0],
    [-1,-1],
    [0,-2],
    [1,-1]
])
DIRECTION_KEYS = (DIRECTIONS+2)[:,0]*5 + (DIRECTIONS+2)[:,1]
DIRECTION_ENCODINGS = np.full(25, -1)
DIRECTION_ENCODINGS[DIRECTION_KEYS] = np.arange(8)

class ChainEncode:
    """Take the output from Marching Squares and chain-encode it to save as compressed npz"""
    @staticmethod
    def chain_encode(shapes, fname="chain_codes.npz"):
        #TODO this compression step would be faster if it was integrated into the working
        # of MarchingSquares itself (e.g., reading the directions as they're written, rather
        # than recalculating them from the outputs. However, the extra computation time likely
        # to be saved is small in comparison to the execution time of MarchingSquares itself.
        #TODO More extensive testing is needed to determine whether 3-bit or 4-bit packing is better
        start_coords = []
        chains = []
        chain_lengths = []
        for shape in shapes:
            shape = np.array(shape)
            step_sizes = shape[1:]-shape[:-1]
            step_keys = (step_sizes+2)[:,0]*5 + (step_sizes+2)[:,1]
            step_codes = DIRECTION_ENCODINGS[step_keys]
            start_coords.append(shape[0])
            chains.append(step_codes)
            chain_lengths.append(len(step_codes))
        concat_chains = np.concatenate(chains)
        np.savez_compressed(
            fname,
            start_coords=np.asarray(start_coords, dtype=np.uint16),
            chains=ChainEncode._pack_4bit(concat_chains),
            chain_lengths=np.asarray(chain_lengths, dtype=np.uint32)
        )

    @staticmethod
    def chain_decode(fname="chain_codes.npz"):
        file = np.load(fname, allow_pickle=True)
        start_coords, packed_concat_chains, chain_lengths = (
            file["start_coords"], file["chains"], file["chain_lengths"]
        )
        concat_chains = ChainEncode._unpack_4bit(packed_concat_chains)

        shapes = []
        pos = 0
        for i, length in enumerate(chain_lengths):
            # Extract single chain
            chain = concat_chains[pos:pos+length]
            pos += length
            # Reconstruct shape
            step_keys = DIRECTION_KEYS[chain]
            step_sizes = np.vstack([step_keys//5,step_keys%5]).T-2
            shape = np.vstack([start_coords[i], np.cumsum(step_sizes, axis=0)+start_coords[i]])
            shapes.append(shape)
        return shapes
    
    @staticmethod
    def _pack_4bit(unpacked):
        unpacked = np.asarray(unpacked, dtype=np.uint8)
        if len(unpacked)%2 != 0:
            unpacked = np.concatenate([unpacked, [15]])
        packed = unpacked[1::2]<<4 | unpacked[::2]
        return packed

    @staticmethod
    def _unpack_4bit(packed):
        evens = packed % (1<<4)
        odds = packed >> 4
        unpacked = np.vstack([evens,odds]).T.flatten()
        if unpacked[-1] == 15:
            return unpacked[:-1]
        return unpacked

    @staticmethod
    def _pack_3bit(unpacked):
        pad_width = (8-len(unpacked))%8
        unpacked = np.concatenate([np.asarray(unpacked, dtype=np.uint32), [0]*pad_width])
        packed_24bit = unpacked[::8]
        for i in range(1, 8):
            bit_offset = i * 3
            packed_24bit += unpacked[i::8] << bit_offset
        highbits = (packed_24bit >> 16) & 0b11111111
        midbits = (packed_24bit >> 8) & 0b11111111
        lowbits = packed_24bit & 0b11111111
        return np.vstack([highbits,midbits,lowbits]).T.flatten()

    @staticmethod
    def _unpack_3bit(packed):
        highbits = packed[::3]
        midbits = packed[1::3]
        lowbits = packed[2::3]
        packed_24bit = lowbits + (midbits<<8) + (highbits<<16)
        unpacked = np.zeros(len(packed_24bit)*8, dtype=np.uint32)
        for i in range(8):
            bit_offset = i * 3
            unpacked[i::8] = (packed_24bit >> bit_offset) & 0b00000111
        return unpacked


if __name__ == "__main__":
    import sys
    sys.path.append("../marching_squares/marching_squares")
    from marching_squares import MarchingSquares
    ms = MarchingSquares()
    print("Performing Marching Squares")
    shapes = MarchingSquares.run("../Dundee.tif", 1/2, render=False)
    print("Done!")
    # Prove compression+decompression is lossless:
    print("Compressing...")
    ChainEncode.chain_encode(shapes, "chain_codes.npz")
    print("Done compressing. Validing compression:")
    print("Decompressing...")
    decoded_shapes = ChainEncode.chain_decode("chain_codes.npz")
    print("Done decompressing, performing validity check:")
    for s1, s2 in zip(shapes, decoded_shapes):
        assert np.all(s1==s2)
    print("All valid.")
