package eu.project.rapid.gvirtus4a;

/*
 * Create the .h for native functions:
 * javah Buffer$Helper
 * 
 * Compile the c files with:
 * gcc -I/usr/lib/jvm/java-1.7.0-openjdk-1.7.0.101.x86_64/include/ 
 * -I/usr/lib/jvm/java-1.7.0-openjdk-1.7.0.101.x86_64/include/linux/ -o libndkBuffer.so -shared -fPIC Buffer.c
 * 
 * Execute the java files with this VM argument:
 * -Djava.library.path=.
 */


import android.os.StrictMode;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

public final class Frontend {
	private static Frontend frontend;
	String serverIpAddress;
	int port;
	Socket socket;
	static DataOutputStream outputStream;
	static DataInputStream in;
	static int resultBufferSize;



	private Frontend(String serverIpAddress, int port) {
		StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
		StrictMode.setThreadPolicy(policy);
		this.serverIpAddress = serverIpAddress;
		this.port = port;
		try {
			socket = new Socket(serverIpAddress, port);
			outputStream = new DataOutputStream(socket.getOutputStream());
			in = new DataInputStream(socket.getInputStream());
		} catch (IOException ex) {
			// TODO gestire la mancata connessione
			throw new RuntimeException(ex);

		}

	}

	public static Frontend getFrontend(String serverIpAddress ,  int port)  {


		if (frontend == null) {
			frontend = new Frontend(serverIpAddress, port);
		}
		return frontend;
	}

	public static int Execute(String routine) throws IOException {


		long size = Buffer.Size() / 2;
		byte[] bits = Util.longToByteArray(size);

        byte[] bytes2 = Util.hexToBytes(Buffer.GetString());


		byte[] outBuffer=new byte[routine.length()+1+bits.length+bytes2.length];

		int j=0;
		for (int i = 0; i < routine.length(); i++) {
			outBuffer[j] = (byte)routine.charAt(i);
			j++;
		}
		outBuffer[j]=0;
        j++;
		for (int i = 0; i < bits.length; i++) {
			outBuffer[j]=(byte) (bits[i] & 0xFF);
            j++;
		}

        for (int i = 0; i < bytes2.length; i++) {
            outBuffer[j]=(byte)(bytes2[i] & 0xFF);
            j++;
        }

		outputStream.write(outBuffer);

		/**************/

		/*
		 //System.out.println("Routine called: " + routine);
		for (int i = 0; i < routine.length(); i++)
			outputStream.writeByte(routine.charAt(i));
		outputStream.writeByte(0);
		long size = Buffer.Size() / 2;
		byte[] bits = Util.longToByteArray(size);

		for (int i = 0; i < bits.length; i++) {
			outputStream.write(bits[i] & 0xFF);
		}

		byte[] bytes2 = Util.hexToBytes(Buffer.GetString());
		for (int i = 0; i < bytes2.length; i++) {
			outputStream.write(bytes2[i] & 0xFF);
		}


		 int message = in.readByte(); // use this for exitcode of single routine
		//in.readByte();
		in.readByte();
		in.readByte();
		in.readByte();

		resultBufferSize = (int) in.readByte();
		for (int i = 0; i < 7; i++)
			in.readByte();
		*/
		byte[] inBuffer=new byte[12];
		in.read(inBuffer,0,12);
		int message=inBuffer[0];
		resultBufferSize=inBuffer[4];
		return message;

	}


	public final static class Transmitter {

		public Transmitter() {

		}

		public void writeLong(DataOutputStream os, long l) throws IOException {
			os.write((byte) l);
			os.write((byte) (l >> 56));
			os.write((byte) (l >> 48));
			os.write((byte) (l >> 40));
			os.write((byte) (l >> 32));
			os.write((byte) (l >> 24));
			os.write((byte) (l >> 16));
			os.write((byte) (l >> 8));
		}

		public void writeChar(DataOutputStream os, char l) throws IOException {
			os.write((byte) l);
			os.write((byte) (l >> 56));
			os.write((byte) (l >> 48));
			os.write((byte) (l >> 40));
			os.write((byte) (l >> 32));
			os.write((byte) (l >> 24));
			os.write((byte) (l >> 16));
			os.write((byte) (l >> 8));
		}

		public void writeInt(DataOutputStream os, int l) throws IOException {
			os.write((byte) l);
			os.write((byte) (l >> 24));
			os.write((byte) (l >> 16));
			os.write((byte) (l >> 8));
		}

		public void writeHex(DataOutputStream os, long x) throws IOException {
			String hex = Integer.toHexString((int) (x));
			StringBuilder out2 = new StringBuilder();
			int scarto = 0;
			if (hex.length() > 2) {
				for (int i = hex.length() - 1; i > 0; i -= 2) {
					String str = hex.substring(i - 1, i + 1);
					out2.insert(0, str);
					os.write((byte) Integer.parseInt(out2.toString(), 16));
					scarto += 2;
				}
				if (scarto != hex.length()) {
					os.write((byte) Integer.parseInt(hex.substring(0, 1), 16));
				}
			}
			os.write((byte) (0));
			os.write((byte) (0));
			os.write((byte) (0));
			os.write((byte) (0));
			os.write((byte) (0));
			os.write((byte) (0));
		}

		public char readChar(DataInputStream os) throws IOException {
			int x;
			x = os.readByte();
			x = x >> 56;
			x = os.readByte();
			x = x >> 48;
			x = os.readByte();
			x = x >> 40;
			x = os.readByte();
			x = x >> 32;
			x = os.readByte();
			x = x >> 24;
			x = os.readByte();
			x = x >> 16;
			x = os.readByte();
			x = x >> 8;
			x = os.readByte();
			return (char) x;

		}

		public static String getHex(int size) throws IOException {
			byte[] array = new byte[size];
			for (int i = 0; i < size; i++) {
				byte bit = in.readByte();
				array[i] = bit;
			}
			String hex = Util.bytesToHex(array);
			return hex;
		}

		public static int getInt() throws IOException {

			StringBuilder output = new StringBuilder();
			for (int i = 0; i < 4; i++) {
				byte bit = in.readByte();
				int a = bit & 0xFF;
				if (a == 0) {
					output.insert(0, Integer.toHexString(a));
					output.insert(0, Integer.toHexString(a));
				} else {
					output.insert(0, Integer.toHexString(a));
				}
			}
			return Integer.parseInt(output.toString(), 16);

		}

		public static long getLong() throws IOException {

			StringBuilder output = new StringBuilder();
			for (int i = 0; i < 8; i++) {
				byte bit = in.readByte();
				int a = bit & 0xFF;
				if (a == 0) {
					output.insert(0, Integer.toHexString(a));
					output.insert(0, Integer.toHexString(a));
				} else {
					output.insert(0, Integer.toHexString(a));
				}
			}
			return Long.parseLong(output.toString(), 16);
		}

		public static float getFloat() throws IOException {
			byte[] inBuffer = new byte[4];
			in.read(inBuffer,0,4);
			return getFloat(inBuffer,0);
		}

		public static float getFloat(byte[] inBuffer,int offset) throws IOException {
			String output = Util.bytesToHex( new byte[]{inBuffer[offset+3],inBuffer[offset+2],inBuffer[offset+1],inBuffer[offset]});
			Long i = Long.parseLong(output, 16);
			Float f = Float.intBitsToFloat(i.intValue());
			return f;
		}

	}
}
