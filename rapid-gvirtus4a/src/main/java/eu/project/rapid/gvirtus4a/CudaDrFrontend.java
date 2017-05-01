package eu.project.rapid.gvirtus4a;

import java.io.IOException;

public class CudaDrFrontend {

	public CudaDrFrontend() {
		Providers providers = Providers.getInstance();
		Provider provider = providers.getBest();
		Frontend.getFrontend(provider.getHost(), provider.getPort());
	}

	public CudaDrFrontend(String serverIpAddress ,  int port) {

		Frontend.getFrontend(serverIpAddress, port);

	}

	public int Execute(String routine) throws IOException {

		int exit_code = Frontend.Execute(routine);
		return exit_code;
	}

	/* CUDA DRIVER DEVICE */
	public int cuDeviceGet(int devID) throws IOException {

		Buffer.clear();
		Buffer.AddPointer(0);
		Buffer.AddInt(devID);
		String outputbuffer = "";
		int exit_c = Execute("cuDeviceGet");
		Util.ExitCode.setExit_code(exit_c);
		int sizeType = Frontend.in.readByte();
		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		for (int i = 0; i < sizeType; i++) {
			if (i == 0 || i == 1) {
				byte bb = Frontend.in.readByte();
				outputbuffer += Integer.toHexString(bb & 0xFF);
			} else
				Frontend.in.readByte();
		}
		StringBuilder out2 = new StringBuilder();
		if (outputbuffer.length() > 2) {
			for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
				String str = outputbuffer.substring(i, i + 2);
				out2.insert(0, str);
			}
			outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
		}
		return Integer.valueOf(outputbuffer);
	}

	public String cuDeviceGetName(int len, int dev) throws IOException {

		Buffer.clear();
		Buffer.AddByte(1);
		for (int i = 0; i < 8; i++)
			Buffer.AddByte(0);
		Buffer.AddByte(1);
		for (int i = 0; i < 7; i++)
			Buffer.AddByte(0);
		Buffer.AddInt(len);
		Buffer.AddInt(dev);

		String outbuffer = "";
		StringBuilder output = new StringBuilder();
		int exit_c = Execute("cuDeviceGetName");
		Util.ExitCode.setExit_code(exit_c);
		int sizeType = Frontend.in.readByte();

		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		Frontend.in.readByte();

		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();

		for (int i = 0; i < sizeType; i++) {
			byte bit = Frontend.in.readByte();
			outbuffer += Integer.toHexString(bit);
		}
		for (int i = 0; i < outbuffer.length() - 1; i += 2) {
			String str = outbuffer.substring(i, i + 2);
			output.append((char) Integer.parseInt(str, 16));

		}
		return output.toString();

	}

	public int cuDeviceGetCount() throws IOException {

		Buffer.clear();
		Buffer.AddPointer(0);
		String outputbuffer = "";
		int exit_c = Execute("cuDeviceGetCount");
		Util.ExitCode.setExit_code(exit_c);
		int sizeType = Frontend.in.readByte();
		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		for (int i = 0; i < sizeType; i++) {
			if (i == 0) {
				byte bb = Frontend.in.readByte();
				outputbuffer += Integer.toHexString(bb & 0xFF);
			} else
				Frontend.in.readByte();
		}
		StringBuilder out2 = new StringBuilder();
		if (outputbuffer.length() > 2) {
			for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
				String str = outputbuffer.substring(i, i + 2);
				out2.insert(0, str);
			}
			outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
		}

		return Integer.valueOf(outputbuffer);

	}

	public int[] cuDeviceComputeCapability(int device) throws IOException {

		Buffer.clear();
		Buffer.AddPointer(0);
		Buffer.AddPointer(0);
		Buffer.AddInt(device);
		String outputbuffer = "";
		int exit_c = Execute("cuDeviceComputeCapability");
		Util.ExitCode.setExit_code(exit_c);
		int sizeType = Frontend.in.readByte();
		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		for (int i = 0; i < sizeType; i++) {
			if (i == 0) {
				byte bb = Frontend.in.readByte();
				outputbuffer += Integer.toHexString(bb & 0xFF);
			} else
				Frontend.in.readByte();
		}
		StringBuilder out2 = new StringBuilder();
		if (outputbuffer.length() > 2) {
			for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
				String str = outputbuffer.substring(i, i + 2);
				out2.insert(0, str);
			}
			outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
		}

		int[] majorminor = new int[2];

		majorminor[0] = Integer.valueOf(outputbuffer);
		outputbuffer = "";
		sizeType = Frontend.in.readByte();
		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		for (int i = 0; i < sizeType; i++) {
			if (i == 0) {
				byte bb = Frontend.in.readByte();
				outputbuffer += Integer.toHexString(bb & 0xFF);
			} else
				Frontend.in.readByte();
		}
		StringBuilder out3 = new StringBuilder();
		if (outputbuffer.length() > 2) {
			for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
				String str = outputbuffer.substring(i, i + 2);
				out3.insert(0, str);
			}
			outputbuffer = String.valueOf(Integer.parseInt(out3.toString(), 16));
		}
		majorminor[1] = Integer.valueOf(outputbuffer);
		return majorminor;

	}

	public int cuDeviceGetAttribute(int attribute, int device) throws IOException {

		Buffer.clear();
		Buffer.AddPointer(0);
		Buffer.AddInt(attribute);
		Buffer.AddInt(device);
		String outputbuffer = "";
		int exit_c = Execute("cuDeviceGetAttribute");
		Util.ExitCode.setExit_code(exit_c);
		int sizeType = Frontend.in.readByte();
		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		for (int i = 0; i < sizeType; i++) {
			if (i == 0) {
				byte bb = Frontend.in.readByte();
				outputbuffer += Integer.toHexString(bb & 0xFF);
			} else
				Frontend.in.readByte();
		}
		StringBuilder out2 = new StringBuilder();
		if (outputbuffer.length() > 2) {
			for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
				String str = outputbuffer.substring(i, i + 2);
				out2.insert(0, str);
			}
			outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
		}

		return Integer.valueOf(outputbuffer);

	}

	public long cuDeviceTotalMem(int dev) throws IOException {

		Buffer.clear();
		Buffer.AddByte(8);
		for (int i = 0; i < 16; i++)
			Buffer.AddByte(0);
		Buffer.AddInt(dev);
		int exit_c = Execute("cuDeviceTotalMem");
		Util.ExitCode.setExit_code(exit_c);
		for (int i = 0; i < 8; i++)
			Frontend.in.readByte();
		long x = Frontend.Transmitter.getLong();
		return x;

	}

	/* CUDA DRIVER MEMORY */

	public String cuMemAlloc(long size) throws IOException {

		Buffer.clear();
		byte[] bits = Util.longToByteArray(size);
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		String pointer = "";
		int exit_c = Execute("cuMemAlloc");
		Util.ExitCode.setExit_code(exit_c);
		pointer = Frontend.Transmitter.getHex(8);
		return pointer;
	}

	public void cuMemcpyHtoD(String dst, float[] src, int count) throws IOException {

		Buffer.clear();
		byte[] bits = Util.longToByteArray(count);
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		Buffer.Add(dst);
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}

		Buffer.Add(src);
		int exit_c = Execute("cuMemcpyHtoD");
		Util.ExitCode.setExit_code(exit_c);


	}

	public float[] cuMemcpyDtoH(String srcDevice, long ByteCount) throws IOException {

		Buffer.clear();
		Buffer.Add(srcDevice);

		byte[] bits = Util.longToByteArray(ByteCount);
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		int exit_c = Execute("cuMemcpyDtoH");
		Util.ExitCode.setExit_code(exit_c);

		int sizeType = (int) ByteCount;// 24576;//98304;
		float[] result = new float[sizeType/4];
		byte[] inBuffer=new byte[sizeType];;
		Frontend.in.read(inBuffer,0,8);

		int bytesToRead=(int)ByteCount;
		int bytesRead;

		int offset = 0;
		do {
			bytesRead = Frontend.in.read(inBuffer,offset,bytesToRead);
			bytesToRead=bytesToRead-bytesRead;
			offset=offset+bytesRead;

		} while (offset<ByteCount);

		int i=0;
		for (offset = 0; offset < sizeType; offset += 4) {
			result[i] = Frontend.Transmitter.getFloat(inBuffer, offset);
			i++;
		}

		return result;

	}

	public void cuMemFree(String ptr) throws IOException {
		Buffer.clear();
		Buffer.Add(ptr);
		int exit_c = Execute("cuMemFree");
		Util.ExitCode.setExit_code(exit_c);

	}

	/* CUDA DRIVER INITIALIZATION */

	public int cuInit(int flags) throws IOException {

		Buffer.clear();
		Buffer.AddInt(flags);
		int exit_c = Execute("cuInit");
		Util.ExitCode.setExit_code(exit_c);
		return 0;
	}

	/* CUDA DRIVER CONTEXT */
	public String cuCtxCreate(int flags, int dev) throws IOException {

		Buffer.clear();
		Buffer.AddInt(flags);
		Buffer.AddInt(dev);
		int exit_c = Execute("cuCtxCreate");
		Util.ExitCode.setExit_code(exit_c);
		return Frontend.Transmitter.getHex(8);
	}

	public void cuCtxDestroy(String ctx) throws IOException {

		Buffer.clear();
		Buffer.Add(ctx);
		int exit_c = Execute("cuCtxDestroy");
		Util.ExitCode.setExit_code(exit_c);
	}

	/* CUDA DRIVER EXECUTION */
	public void cuParamSetv(String hfunc, int offset, String ptr, int numbytes) throws IOException {

		Buffer.clear();
		Buffer.AddInt(offset);
		Buffer.AddInt(numbytes);
		long sizeofp = 8;
		Buffer.Add(sizeofp);
		Buffer.Add(ptr);
		Buffer.Add(hfunc);
		int exit_c = Execute("cuParamSetv");
		Util.ExitCode.setExit_code(exit_c);

	}

	public void cuParamSeti(String hfunc, int offset, int value) throws IOException {

		Buffer.clear();
		Buffer.AddInt(offset);
		Buffer.AddInt(value);
		Buffer.Add(hfunc);
		int exit_c = Execute("cuParamSeti");
		Util.ExitCode.setExit_code(exit_c);
	}

	public void cuParamSetSize(String hfunc, int numbytes) throws IOException {

		Buffer.clear();
		Buffer.AddInt(numbytes);
		Buffer.Add(hfunc);
		int exit_c = Execute("cuParamSetSize");
		Util.ExitCode.setExit_code(exit_c);
	}

	public void cuFuncSetBlockShape(String hfunc, int x, int y, int z) throws IOException {

		Buffer.clear();
		Buffer.AddInt(x);
		Buffer.AddInt(y);
		Buffer.AddInt(z);
		Buffer.Add(hfunc);
		int exit_c = Execute("cuFuncSetBlockShape");
		Util.ExitCode.setExit_code(exit_c);
	}

	public void cuFuncSetSharedSize(String hfunc, int bytes) throws IOException {

		Buffer.clear();
		byte[] bits = Util.intToByteArray(bytes);
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		Buffer.Add(hfunc);
		int exit_c = Execute("cuFuncSetSharedSize");
		Util.ExitCode.setExit_code(exit_c);
	}

	public void cuLaunchGrid(String hfunc, int grid_width, int grid_height) throws IOException {

		Buffer.clear();
		Buffer.AddInt(grid_width);
		Buffer.AddInt(grid_height);
		Buffer.Add(hfunc);
		int exit_c = Execute("cuLaunchGrid");
		Util.ExitCode.setExit_code(exit_c);
	}

	/* CUDA DRIVER MODULE */

	public String cuModuleGetFunction(String cmodule, String str) throws IOException {

		Buffer.clear();
		str = str + "\0";
		long size = str.length();
		byte[] bits = Util.longToByteArray(size);

		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		for (int i = 0; i < size; i++) {
			Buffer.AddByte(str.charAt(i));
		}

		Buffer.Add(cmodule);

		int exit_c = Execute("cuModuleGetFunction");
		Util.ExitCode.setExit_code(exit_c);
		String pointer = "";
		pointer = Frontend.Transmitter.getHex(8);
		for (int i = 0; i < Frontend.resultBufferSize - 8; i++) {
			Frontend.in.readByte();
		}

		return pointer;

	}

	public String cuModuleLoadDataEx(String ptxSource, int jitNumOptions,
			int[] jitOptions, long jitOptVals0, char[] jitOptVals1, long jitOptVals2) throws IOException {
		Buffer.clear();
		Buffer.AddInt(jitNumOptions);
		Buffer.Add(jitOptions);


		// addStringForArgument
		ptxSource = ptxSource + "\0";
		long sizePtxSource = ptxSource.length();

		long size = sizePtxSource;
		byte[] bits = Util.longToByteArray(size);

		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}
		for (int i = 0; i < bits.length; i++) {
			Buffer.AddByte(bits[i] & 0xFF);
		}

		Buffer.AddByte4Ptx(ptxSource, sizePtxSource );

		Buffer.Add(8);
		long OptVals0 = jitOptVals0;
		byte[] bit = Util.longToByteArray(OptVals0);
		for (int i = 0; i < bit.length; i++) {
			Buffer.AddByte(bit[i] & 0xFF);
		}
		Buffer.Add(8);
		Buffer.AddByte(160);
		Buffer.AddByte(159);
		Buffer.AddByte(236);
		Buffer.AddByte(1);
		Buffer.AddByte(0);
		Buffer.AddByte(0);
		Buffer.AddByte(0);
		Buffer.AddByte(0);

		Buffer.Add(8);
		long OptVals2 = jitOptVals2;
		byte[] bit2 = Util.longToByteArray(OptVals2);
		for (int i = 0; i < bit.length; i++) {
			Buffer.AddByte(bit2[i] & 0xFF);
		}

		int exit_c = Execute("cuModuleLoadDataEx");
		Util.ExitCode.setExit_code(exit_c);
		String pointer = "";
		pointer = Frontend.Transmitter.getHex(8);
		for (int i = 0; i < Frontend.resultBufferSize - 8; i++)
			Frontend.in.readByte();

		return pointer;
	}

}
