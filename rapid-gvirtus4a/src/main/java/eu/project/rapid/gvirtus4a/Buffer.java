package eu.project.rapid.gvirtus4a;


import android.util.Log;

public final class Buffer {

    static {
        try {
            System.loadLibrary("native-lib"); // Load native library at runtime
        } catch (UnsatisfiedLinkError e) {
            Log.i("JniTest", "Could not load native library, maybe this is running on the VM.");
        }
    }

    private static String mpBuffer = "";

    public Buffer() {
        mpBuffer = "";
    }

    public static void clear() {
        mpBuffer = "";
    }

    public static void AddPointerNull() {
        byte[] bites = {(byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0};
        mpBuffer += Util.bytesToHex(bites);
    }

    public static void Add(int item) {
        byte[] bites = {(byte) item, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0};
        mpBuffer += Util.bytesToHex(bites);
    }

    public static void Add(long item) {
        byte[] bites = {(byte) item, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 0};
        mpBuffer += Util.bytesToHex(bites);
    }

    public static void Add(String item) {
        byte[] bites = Util.hexToBytes(item);
        mpBuffer += Util.bytesToHex(bites);
    }

    public static void Add(float[] item) {
        String js = prepareFloat(item);  // invoke the native method
        mpBuffer += js;

    }

    public static void Add(int[] item) {
        Add(item.length * 4);
        for (int i = 0; i < item.length; i++) {
            AddInt(item[i]);
        }
    }

    public static void AddInt(int item) {
        byte[] bits = Util.intToByteArray(item);
        mpBuffer += Util.bytesToHex(bits);

    }

    public static void AddPointer(int item) {
        byte[] bites = {(byte) item, (byte) 0, (byte) 0, (byte) 0};
        int size = (Util.Sizeof.INT);
        Add(size);
        mpBuffer += Util.bytesToHex(bites);
    }

    public static String GetString() {
        return mpBuffer;
    }

    public static long Size() {
        return mpBuffer.length();
    }

    public static void AddStruct(CudaDeviceProp struct) {
        byte[] bites = new byte[640];
        bites[0] = (byte) 0x78;
        bites[1] = (byte) 0x02;
        for (int i = 2; i < 640; i++) {
            bites[i] = (byte) 0;

        }
        mpBuffer += Util.bytesToHex(bites);
    }


    public static void AddByte(int i) {
        String jps = prepareSingleByte(i);  // invoke the native method
        mpBuffer += jps;
    }


    public static void AddByte4Ptx(String ptxSource, long size) {
        String jps = preparePtxSource(ptxSource, size);  // invoke the native method
        mpBuffer += jps;
    }

    public static void printMpBuffer() {
        System.out.println("mpBUFFER : " + mpBuffer);
    }

    public static native String prepareFloat(float[] floats);

    public static native String preparePtxSource(String ptxSource, long size);

    public static native String prepareSingleByte(int i);

}
