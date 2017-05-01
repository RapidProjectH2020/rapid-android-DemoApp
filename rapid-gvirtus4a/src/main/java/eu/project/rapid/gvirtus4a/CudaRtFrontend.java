package eu.project.rapid.gvirtus4a;

import java.io.IOException;

public class CudaRtFrontend {
	
	public CudaRtFrontend(String serverIpAddress ,  int port) {

		Frontend.getFrontend(serverIpAddress, port);

	}

	public int Execute(String routine) throws IOException {

		int exit_code = Frontend.Execute(routine);
		return exit_code;

	}

	/* CUDA RUNTIME DEVICE */
	
	public int cudaGetDeviceCount() throws IOException {

		Buffer.clear();
		Buffer.AddPointer(0);
		String outputbuffer = "";
		int exit_c = Execute("cudaGetDeviceCount" );
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
		System.out.println("Integer.valueOf(outputbuffer): " + Integer.valueOf(outputbuffer));
		return Integer.valueOf(outputbuffer);
	}

	public int cudaDeviceCanAccessPeer( int device, int peers) throws IOException {
		Buffer .clear();
		Buffer.AddPointer(0);
		Buffer.AddInt(device);
		Buffer.AddInt(peers);
		String outputbuffer = "";
		int exit_c =  Execute("cudaDeviceCanAccessPeer" );
		//  ExecuteMultiThread("cudaDeviceCanAccessPeer",b, );
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

	public int cudaDriverGetVersion() throws IOException {

		Buffer .clear();
		Buffer.AddPointer(0);
		String outputbuffer = "";
		int exit_c =  Execute("cudaDriverGetVersion" );
		//  ExecuteMultiThread("cudaDriverGetVersion",b, );
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

	public int cudaRuntimeGetVersion() throws IOException {

		Buffer .clear();
		Buffer.AddPointer(0);
		String outputbuffer = "";
		int exit_c =  Execute("cudaRuntimeGetVersion" );
		//  ExecuteMultiThread("cudaRuntimeGetVersion",b, );
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

	public int cudaSetDevice(int device) throws IOException {

		Buffer .clear();
		Buffer.Add(device);
		int exit_c =  Execute("cudaSetDevice" );
		Util.ExitCode.setExit_code(exit_c);
		//  ExecuteMultiThread("cudaSetDevice",b, );
		return 0;
	}

	public String cudaGetErrorString(int error) throws IOException {

		Buffer .clear();
		Buffer.AddInt(error);
		String outbuffer = "";
		StringBuilder output = new StringBuilder();
		int exit_c =  Execute("cudaGetErrorString" );
		Util.ExitCode.setExit_code(exit_c);
		int sizeType = Frontend.in.readByte();
		// System.out.print("sizeType " + sizeType);

		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();
		Frontend.in.readByte();
		// System.out.print("sizeType " + sizeType);

		for (int i = 0; i < 7; i++)
			Frontend.in.readByte();

		for (int i = 0; i < sizeType; i++) {
			byte bit = Frontend.in.readByte();
			outbuffer += Integer.toHexString(bit);
			// System.out.print(outbuffer.toString());
		}
		for (int i = 0; i < outbuffer.length() - 1; i += 2) {
			String str = outbuffer.substring(i, i + 2);
			output.append((char) Integer.parseInt(str, 16));

		}
		return output.toString();

	}

	public void cudaDeviceReset() throws IOException {
		Buffer .clear();
		int exit_c =  Execute("cudaDeviceReset" );
		Util.ExitCode.setExit_code(exit_c);

	}

	public CudaDeviceProp cudaGetDeviceProperties( int device) throws IOException {
		Buffer.clear();
		String outbuffer = "";
		StringBuilder output = new StringBuilder();
		CudaDeviceProp struct = new CudaDeviceProp();

		Buffer.AddStruct(struct);
		Buffer.AddInt(device);
		int exit_c =  Execute("cudaGetDeviceProperties" );
		Util.ExitCode.setExit_code(exit_c);
		for (int i = 0; i < 8; i++) {
			Frontend.in.readByte();
		}
		for (int i = 0; i < 256; i++) {
			byte bit = Frontend.in.readByte();

			outbuffer += Integer.toHexString(bit);
		}
		for (int i = 0; i < outbuffer.length() - 1; i += 2) {
			String str = outbuffer.substring(i, i + 2);
			output.append((char) Integer.parseInt(str, 16));
		}
		struct.setName(output.toString());
		struct.setTotalGlobalMem(Frontend.Transmitter.getLong());
		struct.setSharedMemPerBlock(Frontend.Transmitter.getLong());
		struct.setRegsPerBlock(Frontend.Transmitter.getInt());
		
		struct.setWarpSize(Frontend.Transmitter.getInt());
		struct.setMemPitch(Frontend.Transmitter.getLong());
		struct.setMaxThreadsPerBlock(Frontend.Transmitter.getInt());
		struct.setMaxThreadsDim(Frontend.Transmitter.getInt(),0);
		
		
		struct.setMaxThreadsDim(Frontend.Transmitter.getInt(),1);
		
		struct.setMaxThreadsDim(Frontend.Transmitter.getInt(), 2);
		struct.setMaxGridSize(Frontend.Transmitter.getInt(),0);
		struct.setMaxGridSize(Frontend.Transmitter.getInt(),1);
		struct.setMaxGridSize(Frontend.Transmitter.getInt(),2);
		struct.setClockRate(Frontend.Transmitter.getInt()); // check
		struct.setTotalConstMem(Frontend.Transmitter.getLong());
		struct.setMajor(Frontend.Transmitter.getInt());
		struct.setMinor(Frontend.Transmitter.getInt());
		struct.setTextureAlignment(Frontend.Transmitter.getLong());
		struct.setTexturePitchAlignment(Frontend.Transmitter.getLong()); // check
		struct.setDeviceOverlap(Frontend.Transmitter.getInt());
		struct.setMultiProcessorCount(Frontend.Transmitter.getInt());
		struct.setKernelExecTimeoutEnabled(Frontend.Transmitter.getInt());
		struct.setIntegrated(Frontend.Transmitter.getInt());
		struct.setCanMapHostMemory(Frontend.Transmitter.getInt());
		struct.setComputeMode(Frontend.Transmitter.getInt());
		struct.setMaxTexture1D(Frontend.Transmitter.getInt());
		struct.setMaxTexture1DMipmap(Frontend.Transmitter.getInt());
		struct.setMaxTexture1DLinear(Frontend.Transmitter.getInt()); // check
		struct.setMaxTexture2D(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture2D(Frontend.Transmitter.getInt(),1);
		
		struct.setMaxTexture2DMipmap(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture2DMipmap(Frontend.Transmitter.getInt(),1);
		
		struct.setMaxTexture2DLinear(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture2DLinear(Frontend.Transmitter.getInt(),1);
		struct.setMaxTexture2DLinear(Frontend.Transmitter.getInt(),2);
		
		struct.setMaxTexture2DGather(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture2DGather(Frontend.Transmitter.getInt(),1);
		
		struct.setMaxTexture3D(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture3D(Frontend.Transmitter.getInt(),1);
		struct.setMaxTexture3D(Frontend.Transmitter.getInt(),2);
		
		struct.setMaxTexture3DAlt(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture3DAlt(Frontend.Transmitter.getInt(),1);
		struct.setMaxTexture3DAlt(Frontend.Transmitter.getInt(),2);
		struct.setMaxTextureCubemap(Frontend.Transmitter.getInt());
		struct.setMaxTexture1DLayered(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture1DLayered(Frontend.Transmitter.getInt(),1);
		struct.setMaxTexture2DLayered(Frontend.Transmitter.getInt(),0);
		struct.setMaxTexture2DLayered(Frontend.Transmitter.getInt(),1);
		struct.setMaxTexture2DLayered(Frontend.Transmitter.getInt(),2);
		struct.setMaxTextureCubemapLayered(Frontend.Transmitter.getInt(),0);
		struct.setMaxTextureCubemapLayered(Frontend.Transmitter.getInt(),1);
		struct.setMaxSurface1D(Frontend.Transmitter.getInt());
		struct.setMaxSurface2D(Frontend.Transmitter.getInt(),0);
		struct.setMaxSurface2D(Frontend.Transmitter.getInt(),1);
		struct.setMaxSurface3D(Frontend.Transmitter.getInt(),0);
		struct.setMaxSurface3D(Frontend.Transmitter.getInt(),1);
		struct.setMaxSurface3D(Frontend.Transmitter.getInt(),2);
		struct.setMaxSurface1DLayered(Frontend.Transmitter.getInt(),0);
		struct.setMaxSurface1DLayered(Frontend.Transmitter.getInt(),1);
		struct.setMaxSurface2DLayered(Frontend.Transmitter.getInt(),0);
		struct.setMaxSurface2DLayered(Frontend.Transmitter.getInt(),1);
		struct.setMaxSurface2DLayered(Frontend.Transmitter.getInt(),2);
		struct.setMaxSurfaceCubemap(Frontend.Transmitter.getInt());
		struct.setMaxSurfaceCubemapLayered(Frontend.Transmitter.getInt(),0);
		struct.setMaxSurfaceCubemapLayered(Frontend.Transmitter.getInt(),1);
		struct.setSurfaceAlignment(Frontend.Transmitter.getLong());
		struct.setConcurrentKernels(Frontend.Transmitter.getInt());
		struct.setECCEnabled(Frontend.Transmitter.getInt());
		struct.setPciBusID(Frontend.Transmitter.getInt());
		struct.setPciDeviceID(Frontend.Transmitter.getInt());
		struct.setPciDomainID(Frontend.Transmitter.getInt());
		struct.setTccDriver(Frontend.Transmitter.getInt());
		struct.setAsyncEngineCount(Frontend.Transmitter.getInt());
		struct.setUnifiedAddressing(Frontend.Transmitter.getInt());
		struct.setMemoryClockRate(Frontend.Transmitter.getInt());
		struct.setMemoryBusWidth(Frontend.Transmitter.getInt());
		struct.setL2CacheSize(Frontend.Transmitter.getInt());
		struct.setMaxThreadsPerMultiProcessor(Frontend.Transmitter.getInt());
		struct.setStreamPrioritiesSupported(Frontend.Transmitter.getInt());
		struct.setGlobalL1CacheSupported(Frontend.Transmitter.getInt());
		struct.setLocalL1CacheSupported(Frontend.Transmitter.getInt());
		struct.setSharedMemPerMultiprocessor(Frontend.Transmitter.getLong());
		struct.setRegsPerMultiprocessor(Frontend.Transmitter.getInt());
		struct.setManagedMemory(Frontend.Transmitter.getInt());
		struct.setIsMultiGpuBoard(Frontend.Transmitter.getInt());
		struct.setMultiGpuBoardGroupID(Frontend.Transmitter.getInt());
		Frontend.Transmitter.getInt(); // è in più da capire il perchè
		return struct;
	}
}
