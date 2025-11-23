import numpy as np

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


if __name__ == "__main__":
	from ..marching_squares.marching_squares import MarchingSquares
	ms = MarchingSquares()
	shapes = MarchingSquares.run("../Dundee.tif", 1, render=False)
	# Prove compression+decompression is lossless:
	ChainEncode.chain_encode(shapes, "chain_codes.npz")
	decoded_shapes = ChainEncode.chain_decode("chain_codes.npz")
	for s1, s2 in zip(shapes, decoded_shapes):
		assert np.all(s1==s2)
