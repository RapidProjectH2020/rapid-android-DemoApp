package eu.project.rapid.gvirtus;

import android.util.Log;

import java.io.IOException;
import java.lang.reflect.Method;

import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.Remote;
import eu.project.rapid.ac.Remoteable;
import eu.project.rapid.gvirtus4a.CudaDrFrontend;
import eu.project.rapid.gvirtus4a.CudaException;
import eu.project.rapid.gvirtus4a.Provider;
import eu.project.rapid.gvirtus4a.Providers;
import eu.project.rapid.gvirtus4a.Util;
import eu.project.rapid.gvirtus4a.params.FloatArrayParam;
import eu.project.rapid.gvirtus4a.params.IntParam;
import eu.project.rapid.gvirtus4a.params.StringParam;

/**
 * Created by raffaelemontella on 26/04/2017.
 */

public class MatrixMul extends Remoteable {
    private transient DFE dfe;

    private int widthA;
    private int heightA;
    private int widthB;

    private String ptxSource;
    private static final String TAG = "MatrixMul";

    public MatrixMul(DFE dfe) {
        this.dfe = dfe;
        String ptxName = "cuda-kernels/matrixMul_kernel64.ptx";
        try {
            ptxSource = Util.readAssetFileAsString(dfe.getContext(), ptxName);
            Log.v(TAG, "Read the PTX source");
            Log.v(TAG, ptxSource);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    @Override
    public void prepareDataOnClient() {

    }

    @Override
    public void copyState(Remoteable state) {

    }

    public boolean gpuMatrixMul(int widthA, int heightA, int widthB) {
        this.widthA = widthA;
        this.heightA = heightA;
        this.widthB = widthB;
        Method toExecute;
        boolean result = false;
        Class<?>[] paramTypes = {int.class, int.class, int.class};
        Object[] paramValues = {widthA, heightA, widthB};

        try {
            toExecute = this.getClass().getDeclaredMethod("localGpuMatrixMul", paramTypes);
            result = (Boolean) dfe.execute(toExecute, paramValues, this);
        } catch (SecurityException e) {
            // Should never get here
            e.printStackTrace();
            throw e;
        } catch (NoSuchMethodException e) {
            // Should never get here
            e.printStackTrace();
        } catch (Throwable e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return result;
    }

    @Remote
    public boolean localGpuMatrixMul(int widthA, int heightA, int widthB) {
        Log.v(TAG, "Entered matrixMul");

        final float valB = 0.01f;
        int exit_c=0;
        //CudaDrFrontend driver = new CudaDrFrontend("193.205.230.23", 9998);
        Provider bestProvider=Providers.getInstance().getDefaultProvider();
        try {
            CudaDrFrontend driver = new CudaDrFrontend(bestProvider.getHost(),bestProvider.getPort());


            exit_c=driver.cuInit(0);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            StringParam spCuContext = new StringParam();

            exit_c=driver.cuCtxCreate(0, 0, spCuContext);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            String cuContext = spCuContext.value;

            IntParam ipDevice = new IntParam();
            exit_c=driver.cuDeviceGet(0, ipDevice);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }

            Log.v(TAG, "matrixMul 1");

            int jitNumOptions = 3;
            int[] jitOptions = new int[jitNumOptions];

            // set up size of compilation log buffer
            jitOptions[0] = 4;// CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
            long jitLogBufferSize = 1024;
            long jitOptVals0 = jitLogBufferSize;

            // set up pointer to the compilation log buffer
            jitOptions[1] = 3;// CU_JIT_INFO_LOG_BUFFER;

            char[] jitLogBuffer = new char[(int) jitLogBufferSize];
            char[] jitOptVals1 = jitLogBuffer;

            Log.v(TAG, "matrixMul 2");

            // set up pointer to set the Maximum # of registers for a particular
            // kernel
            jitOptions[2] = 0;// CU_JIT_MAX_REGISTERS;
            long jitRegCount = 32;
            long jitOptVals2 = jitRegCount;

            Log.v(TAG, "matrixMul 3");

            StringParam spModule = new StringParam();
            exit_c=driver.cuModuleLoadDataEx(
                                ptxSource, jitNumOptions, jitOptions, jitOptVals0,
                                jitOptVals1, jitOptVals2, spModule);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            String cmodule = spModule.value;
            Log.v(TAG, "matrixMul 4");

            StringParam spFunction = new StringParam();
            exit_c=driver.cuModuleGetFunction(cmodule, "matrixMul_bs32_32bit", spFunction);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            String cfunction = spFunction.value;
            Log.v(TAG, "matrixMul 5");

            // allocate host memory for matrices A and B
            int block_size = 32; // larger block size is for Fermi and above
            final int WA = (widthA * block_size); // Matrix A width
            final int HA = (heightA * block_size); // Matrix A height
            final int WB = (widthB * block_size); // Matrix B width
            final int HB = WA; // Matrix B height
            int WC = WB; // Matrix C width
            int HC = HA; // Matrix C height

            int size_A = WA * HA;
            int mem_size_A = Float.SIZE / 8 * size_A;
            float[] h_A = new float[size_A];
            int size_B = WB * HB;
            int mem_size_B = Float.SIZE / 8 * size_B;
            float[] h_B = new float[size_B];
            //System.out.prinf("%.2f", valB);

            Log.v(TAG, "matrixMul 6");

            h_A = constantInit(h_A, size_A, 1.0f);
            h_B = constantInit(h_B, size_B, valB);
            // allocate device memory

            StringParam spD_A = new StringParam();
            exit_c=driver.cuMemAlloc(mem_size_A, spD_A);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            String d_A = spD_A.value;

            StringParam spD_B = new StringParam();
            exit_c=driver.cuMemAlloc(mem_size_B, spD_B);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            String d_B = spD_B.value;

            driver.cuMemcpyHtoD(d_A, h_A, mem_size_A);
            driver.cuMemcpyHtoD(d_B, h_B, mem_size_B);
            // allocate device memory for result
            long size_C = WC * HC;
            float[] h_C;

            Log.v(TAG, "matrixMul 7");

            long mem_size_C = Float.SIZE / 8 * size_C;

            StringParam spD_C = new StringParam();
            exit_c=driver.cuMemAlloc(mem_size_C, spD_C);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            String d_C = spD_C.value;

            Util.Dim3 grid = new Util.Dim3(WC / block_size, HC / block_size, 1);

            int offset = 0;
            // setup execution parameters

            Log.v(TAG, "matrixMul 8");

            driver.cuParamSetv(cfunction, offset, d_C, Util.Sizeof.LONG);

            offset += Util.Sizeof.LONG;
            driver.cuParamSetv(cfunction, offset, d_A, Util.Sizeof.LONG);
            offset += Util.Sizeof.LONG;
            driver.cuParamSetv(cfunction, offset, d_B, Util.Sizeof.LONG);
            offset += Util.Sizeof.LONG;

            Log.v(TAG, "matrixMul 9");

            int Matrix_Width_A = WA;
            int Matrix_Width_B = WB;
            int Sizeof_Matrix_Width_A = Util.Sizeof.INT;
            int Sizeof_Matrix_Width_B = Util.Sizeof.INT;


            driver.cuParamSeti(cfunction, offset, Matrix_Width_A);

            Log.v(TAG, "matrixMul 10");

            offset += Sizeof_Matrix_Width_A;
            driver.cuParamSeti(cfunction, offset, Matrix_Width_B);
            offset += Sizeof_Matrix_Width_B;

            Log.v(TAG, "matrixMul 11");

            driver.cuParamSetSize(cfunction, offset);
            driver.cuFuncSetBlockShape(cfunction, block_size, block_size, grid.z);
            driver.cuFuncSetSharedSize(cfunction, 2 * block_size * block_size * (Float.SIZE / 8));
            driver.cuLaunchGrid(cfunction, grid.x, grid.y);

            FloatArrayParam fapH_C = new FloatArrayParam();
            exit_c=driver.cuMemcpyDtoH(d_C, mem_size_C, fapH_C);
            if (exit_c != 0) {
                throw new RuntimeException(new CudaException(exit_c));
            }
            h_C = fapH_C.values;

            Log.v(TAG, "matrixMul 12");

            boolean correct = true;
            for (int i = 0; i < WC * HC; i++) {
                if (Math.abs(h_C[i] - (WA * valB)) > 1e-2) {
                    correct = false;
                }
            }

            Log.v(TAG, "matrixMul 13");

            driver.cuMemFree(d_A);
            driver.cuMemFree(d_B);
            driver.cuMemFree(d_C);
            driver.cuCtxDestroy(cuContext);

            driver.close();
            Log.v(TAG, "matrixMul 14");
            return correct;

        } catch (IOException ex) {
            Log.e(TAG, "Error while running MatrixMul: " + ex);
            throw new RuntimeException(ex);
        }
    }

    public static float[][] makeMatrix(int dim1, int dim2, float valB) {
        float[][] matrix = new float[dim1][dim2];
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j < matrix[i].length; j++)
                matrix[i][j] = valB;
        return matrix;
    }

    public static float[] constantInit(float[] data, int size, float val) {
        for (int i = 0; i < size; ++i) {
            data[i] = val;
        }
        return data;
    }
}
