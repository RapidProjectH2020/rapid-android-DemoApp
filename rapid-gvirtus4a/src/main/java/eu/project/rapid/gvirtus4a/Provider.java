package eu.project.rapid.gvirtus4a;

import java.io.IOException;
import java.util.Vector;

/**
 * Created by raffaelemontella on 26/04/2017.
 */

public class Provider {
    private String host;
    private int port;
    private int driverVersion;
    private int runtimeVersion;

    private Vector<CudaDeviceProp> properties=new Vector<CudaDeviceProp>();

    public Provider(String host, int port) {
        this.host=host;
        this.port=port;

        try {
            deviceQuery();
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    public String getHost() { return host; }
    public int getPort() { return port; }

    public void deviceQuery() throws IOException {

        CudaRtFrontend runtime = new CudaRtFrontend(host, port);
        int deviceCount = runtime.cudaGetDeviceCount();
        if (Util.ExitCode.getExit_code() != 0) {
            return;
        }

        driverVersion = runtime.cudaDriverGetVersion();
        runtimeVersion = runtime.cudaRuntimeGetVersion();

        for (int i = 0; i < deviceCount; i++) {
            runtime.cudaSetDevice(i);
            CudaDeviceProp deviceProp = runtime.cudaGetDeviceProperties(i);
            properties.add(deviceProp);
        }
    }
}
