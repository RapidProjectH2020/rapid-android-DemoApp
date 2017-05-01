/*******************************************************************************
 * Copyright (C) 2015, 2016 RAPID EU Project
 *
 * This library is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either version
 * 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA
 *******************************************************************************/
package eu.project.rapid.ac;

import android.app.IntentService;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.preference.PreferenceManager;
import android.util.Log;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.DatagramPacket;
import java.net.InetSocketAddress;
import java.net.MulticastSocket;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import eu.project.rapid.ac.d2d.D2DMessage;
import eu.project.rapid.ac.d2d.D2DMessage.MsgType;
import eu.project.rapid.ac.d2d.PhoneSpecs;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidConstants;
import eu.project.rapid.common.RapidMessages;

/**
 * This thread will be started by clients that run the DFE so that these clients can get the HELLO
 * messages sent by the devices that act as D2D Acceleration Server.
 *
 * @author sokol
 */
public class RapidNetworkService extends IntentService {

    private static final String TAG = RapidNetworkService.class.getName();
    ScheduledThreadPoolExecutor setWriterScheduledPool =
            (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(1);
    static final int FREQUENCY_WRITE_D2D_SET = 5 * 60 * 1013; // Every 5 minutes save the set
    public static final int FREQUENCY_READ_D2D_SET = 1 * 60 * 1011; // Every 1 minute read the set
    private Set<PhoneSpecs> setD2dPhones = new TreeSet<>(); // Sorted by specs

    ScheduledThreadPoolExecutor netScheduledPool =
            (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(2);
    // Every 30 minutes measure rtt, ulRate, and dlRate
    static final int FREQUENCY_NET_MEASUREMENT = 30 * 60 * 1000;

    private Configuration config;
    private boolean usePrevVm = true;
    private Clone sClone;
    private long myId = -1;
    private String vmIp = "";
    private ArrayList<String> vmmIPs;
    private static final int vmNrVCPUs = 1; // FIXME: number of CPUs on the VM
    private static final int vmMemSize = 512; // FIXME
    private static final int vmNrGpuCores = 1200; // FIXME

    // Intent for sending broadcast messages
    public static final String RAPID_VM_CHANGED = "eu.project.rapid.vmChanged";
    public static final String RAPID_VM_IP = "eu.project.rapid.vmIP";
    public static final String RAPID_NETWORK_CHANGED = "eu.project.rapid.networkChanged";
    public static final String RAPID_NETWORK_RTT = "eu.project.rapid.rtt";
    public static final String RAPID_NETWORK_UL_RATE = "eu.project.rapid.ulRate";
    public static final String RAPID_NETWORK_DL_RATE = "eu.project.rapid.dlRate";

    public RapidNetworkService() {
        super(RapidNetworkService.class.getName());
    }

    @Override
    protected void onHandleIntent(Intent intent) {
        Log.d(TAG, "Started the service");

        try (ServerSocket acRmServerSocket = new ServerSocket(23456)) {
            Log.d(TAG, "************* Started the AC_RM listening server ****************");
            readConfigurationFile();
            // The prev id is useful to the DS so that it can release already allocated VMs.
            SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
            myId = prefs.getLong(Constants.MY_OLD_ID, -1);
            vmIp = prefs.getString(Constants.PREV_VM_IP, "");
            String vmmIp = prefs.getString(Constants.PREV_VMM_IP, "");

            new Thread(new D2DListener()).start();

            setWriterScheduledPool.scheduleWithFixedDelay(new D2DSetWriter(), FREQUENCY_WRITE_D2D_SET,
                    FREQUENCY_WRITE_D2D_SET, TimeUnit.MILLISECONDS);

            netScheduledPool.scheduleWithFixedDelay(new NetMeasurementRunnable(), 10,
                    FREQUENCY_NET_MEASUREMENT, TimeUnit.MILLISECONDS);


            registerWithDsAndSlam();
            while (true) {
                try (Socket client = acRmServerSocket.accept()) {
                    new Thread(new ClientHandler(client)).start();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            Log.w(TAG, "Couldn't start the listening server, " +
                    "maybe another app has already started this service: " + e);
        }
    }

    private void readConfigurationFile() {
        try {
            // Read the config file to read the IP and port of Manager
            config = new Configuration(Constants.PHONE_CONFIG_FILE);
            config.parseConfigFile();
        } catch (FileNotFoundException e) {
            Log.e(TAG, "Config file not found: " + Constants.PHONE_CONFIG_FILE);
            config = new Configuration();
        }
    }

    private class D2DListener implements Runnable {
        private final String TAG = D2DListener.class.getName();

        @Override
        public void run() {

            try {
                Log.i(TAG, "Thread started");
                writeSetOnFile();

                WifiManager.MulticastLock lock = null;
                WifiManager wifi = (WifiManager) getSystemService(Context.WIFI_SERVICE);
                Log.i(TAG, "Trying to acquire multicast lock...");
                if (Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.KITKAT) {
                    if (wifi != null) {
                        lock = wifi.createMulticastLock("WiFi_Lock");
                        lock.setReferenceCounted(true);
                        lock.acquire();
                        Log.i(TAG, "Lock acquired!");
                    }
                }

                MulticastSocket receiveSocket = new MulticastSocket(Constants.D2D_BROADCAST_PORT);
                receiveSocket.setBroadcast(true);
                Log.i(TAG, "Started listening on multicast socket.");

                try {
                    // This will be interrupted when the OS kills the service
                    while (true) {
                        Log.i(TAG, "Waiting for broadcasted data...");
                        byte[] data = new byte[1024];
                        DatagramPacket packet = new DatagramPacket(data, data.length);
                        receiveSocket.receive(packet);
                        Log.d(TAG, "Received a new broadcast packet from: " + packet.getAddress());
                        processPacket(packet);
                    }
                } catch (IOException e) {
                    Log.d(TAG, "The socket was closed.");
                }

                Log.i(TAG, "Stopped receiving data!");
            } catch (IOException e) {
                // We expect this to happen when more than one DFE on the same phone will try to create
                // this service and the port will be busy. This way only one service will be listening for D2D
                // messages. This service will be responsible for writing the received messages on a file so
                // that the DFEs of all applications could read them.
                Log.d(TAG,
                        "Could not create D2D multicast socket, maybe the service is already started by another DFE: "
                                + e);
                // e.printStackTrace();
            }
        }
    }


    /**
     * Process the packet received by another device in a D2D scenario. Create a D2Dmessage and if
     * this is a HELLO message then store the specifics of the other device into the Map. If a new
     * device is added to the map and more than 5 minutes have passed since the last time we saved the
     * devices on the file, then save the set in the filesystem so that other DFEs can read it.
     *
     * @param packet
     */
    private void processPacket(DatagramPacket packet) {
        try {
            D2DMessage msg = new D2DMessage(packet.getData());
            Log.d(TAG, "Received: <== " + msg);
            if (msg.getMsgType() == MsgType.HELLO) {
                PhoneSpecs otherPhone = msg.getPhoneSpecs();
                if (setD2dPhones.contains(otherPhone)) {
                    setD2dPhones.remove(otherPhone);
                }
                otherPhone.setTimestamp(System.currentTimeMillis());
                otherPhone.setIp(packet.getAddress().getHostAddress());
                setD2dPhones.add(otherPhone);
                // FIXME writing the set here is too heavy but I want this just for the demo. Later fix this
                // with a smarter alternative.
                writeSetOnFile();
            }
        } catch (IOException | ClassNotFoundException e) {
            Log.e(TAG, "Error while processing the packet: " + e);
        }
    }

    private class D2DSetWriter implements Runnable {
        @Override
        public void run() {
            // Write the set in the filesystem so that other DFEs can use the D2D phones when needed.
            Iterator<PhoneSpecs> it = setD2dPhones.iterator();
            // First clean the set from devices that have not been pinging recently
            while (it.hasNext()) {
                // If the last time we have seen this device is 5 pings before, then remove it.
                if ((System.currentTimeMillis() - it.next().getTimestamp()) > 5
                        * Constants.D2D_BROADCAST_INTERVAL) {
                    it.remove();
                }
            }
            writeSetOnFile();
        }
    }

    private void writeSetOnFile() {
        try {
            Log.i(TAG, "Writing set of D2D devices on the sdcard file");
            // This method is blocking, waiting for the lock on the file to be available.
            Utils.writeObjectToFile(Constants.FILE_D2D_PHONES, setD2dPhones);
            Log.i(TAG, "Finished writing set of D2D devices on the sdcard file");
        } catch (IOException e) {
            Log.e(TAG, "Error while writing set of D2D devices on the sdcard file: " + e);
        }
    }

    private class NetMeasurementRunnable implements Runnable {
        @Override
        public void run() {
            int maxWaitTime = 5 * 60 * 1000; // 5 min
            int waitingSoFar = 0;

            while (sClone == null && waitingSoFar < maxWaitTime) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                waitingSoFar += 1000;
            }

            if (sClone != null) {
                NetworkProfiler.measureRtt(sClone.getIp(), config.getClonePortBandwidthTest());
                NetworkProfiler.measureUlRate(sClone.getIp(), config.getClonePortBandwidthTest());
                NetworkProfiler.measureDlRate(sClone.getIp(), config.getClonePortBandwidthTest());

                Intent intent = new Intent(RapidNetworkService.RAPID_NETWORK_CHANGED);
                intent.putExtra(RapidNetworkService.RAPID_NETWORK_RTT, NetworkProfiler.rtt);
                intent.putExtra(RapidNetworkService.RAPID_NETWORK_DL_RATE, NetworkProfiler.lastDlRate.getBw());
                intent.putExtra(RapidNetworkService.RAPID_NETWORK_UL_RATE, NetworkProfiler.lastUlRate.getBw());
                sendBroadcast(intent);
            }
        }
    }

    private class ClientHandler implements Runnable {
        private Socket clientSocket;

        ClientHandler(Socket clientSocket) {
            this.clientSocket = clientSocket;
        }

        @Override
        public void run() {
            try (InputStream is = clientSocket.getInputStream();
                 OutputStream os = clientSocket.getOutputStream();
                 ObjectInputStream ois = new ObjectInputStream(is);
                 ObjectOutputStream oos = new ObjectOutputStream(os)) {

                int command = is.read();
                switch (command) {
                    case 1:
                        oos.writeObject(sClone);
                        oos.flush();
                        break;

                    case 2:
                        oos.writeInt(NetworkProfiler.lastUlRate.getBw());
                        oos.writeInt(NetworkProfiler.lastDlRate.getBw());
                        oos.writeInt(NetworkProfiler.rtt);
                        oos.flush();
                        break;

                    default:
                        Log.w(TAG, "Did not recognize command: " + command);
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private boolean registerWithDsAndSlam() {
        Log.i(TAG, "Registering...");
        boolean registeredWithSlam = false;

        if (registerWithDs()) {
            // register with SLAM. Send the VMM where we want to start the VM
            int vmmIndex = 0;
            if (vmmIPs != null) {
                do {
                    registeredWithSlam = registerWithSlam(vmmIPs.get(vmmIndex));
                    vmmIndex++;
                } while (!registeredWithSlam && vmmIndex < vmmIPs.size());
            }
        }
        return registeredWithSlam;
    }

    /**
     * Read the config file to get the IP and port of the DS. The DS will return a list of available
     * SLAMs, choose the best one from the list and connect to it to ask for a VM.
     */
    @SuppressWarnings("unchecked")
    private boolean registerWithDs() {

        Log.d(TAG, "Starting as phone with ID: " + myId);

        Socket dsSocket = new Socket();
        boolean connectedWithDs = false;
        do {
            Log.i(TAG, "Registering with DS " + config.getDSIp() + ":" + config.getDSPort());
            try {
                dsSocket.connect(new InetSocketAddress(config.getDSIp(), config.getDSPort()), 5000);
                Log.d(TAG, "Connected with DS");
                connectedWithDs = true;
            } catch (Exception e) {
                Log.e(TAG, "Could not connect with the DS: " + e);
                try {
                    Thread.sleep(10 * 1000);
                } catch (InterruptedException e1) {
                    Thread.currentThread().interrupt();
                }
            }
        } while (!connectedWithDs);

        try (ObjectOutputStream dsOut = new ObjectOutputStream(dsSocket.getOutputStream());
             ObjectInputStream dsIn = new ObjectInputStream(dsSocket.getInputStream())) {

            // Send the name and id to the DS
            if (usePrevVm) {
                Log.i(TAG, "AC_REGISTER_PREV_DS");
                // Send message format: command (java byte), userId (java long), qosFlag (java int)
                dsOut.writeByte(RapidMessages.AC_REGISTER_PREV_DS);
                dsOut.writeLong(myId); // send my user ID so that my previous VM can be released
            } else { // Connect to a new VM
                Log.i(TAG, "AC_REGISTER_NEW_DS");
                dsOut.writeByte(RapidMessages.AC_REGISTER_NEW_DS);

                dsOut.writeLong(myId); // send my user ID so that my previous VM can be released
                // FIXME: should not use hard-coded values here.
                dsOut.writeInt(vmNrVCPUs); // send vcpuNum as int
                dsOut.writeInt(vmMemSize); // send memSize as int
                dsOut.writeInt(vmNrGpuCores); // send gpuCores as int
            }

            dsOut.flush();

            // Receive message format: status (java byte), userId (java long), SLAM ipAddress (java UTF)
            byte status = dsIn.readByte();
            Log.i(TAG, "Return Status: " + (status == RapidMessages.OK ? "OK" : "ERROR"));
            if (status == RapidMessages.OK) {
                myId = dsIn.readLong();
                Log.i(TAG, "userId is: " + myId);

                // Read the list of VMMs, which will be sorted based on free resources
                vmmIPs = (ArrayList<String>) dsIn.readObject();

                // Read the SLAM IP and port
                String slamIp = dsIn.readUTF();
                int slamPort = dsIn.readInt();
                config.setSlamIp(slamIp);
                config.setSlamPort(slamPort);
                Log.i(TAG, "SLAM address is: " + slamIp + ":" + slamPort);

                return true;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error while connecting with the DS: " + e);
        }

        return false;
    }

    /**
     * Register with the SLAM.
     */
    private boolean registerWithSlam(String vmmIp) {

        Socket slamSocket = new Socket();
        boolean connectedWithSlam = false;
        do {
            Log.i(TAG, "Registering with SLAM " + config.getSlamIp() + ":" + config.getSlamPort());
            try {
                slamSocket.connect(new InetSocketAddress(config.getSlamIp(), config.getSlamPort()), 5000);
                Log.d(TAG, "Connected with SLAM");
                connectedWithSlam = true;
            } catch (Exception e) {
                Log.e(TAG, "Could not connect with the SLAM: " + e);
                try {
                    Thread.sleep(10 * 1000);
                } catch (InterruptedException e1) {
                    Thread.currentThread().interrupt();
                }
            }
        } while (!connectedWithSlam);

        try (ObjectOutputStream oos = new ObjectOutputStream(slamSocket.getOutputStream());
             ObjectInputStream ois = new ObjectInputStream(slamSocket.getInputStream())) {

            // Send the ID to the SLAM
            oos.writeByte(RapidMessages.AC_REGISTER_SLAM);
            oos.writeLong(myId);
            oos.writeInt(RapidConstants.OS.ANDROID.ordinal());

            // Send the vmmId and vmmPort to the SLAM so it can start the VM
            oos.writeUTF(vmmIp);
            oos.writeInt(config.getVmmPort());

            // FIXME: should not use hard-coded values here.
            oos.writeInt(vmNrVCPUs); // send vcpuNum as int
            oos.writeInt(vmMemSize); // send memSize as int
            oos.writeInt(vmNrGpuCores); // send gpuCores as int

            oos.flush();

            int slamResponse = ois.readByte();
            if (slamResponse == RapidMessages.OK) {
                Log.i(TAG, "SLAM OK, getting the VM details");
                vmIp = ois.readUTF();

                sClone = new Clone("", vmIp);
                sClone.setId((int) myId);

                Log.i(TAG, "Saving my ID and the vmIp: " + myId + ", " + vmIp);
                SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
                SharedPreferences.Editor editor = prefs.edit();

                editor.putLong(Constants.MY_OLD_ID, myId);
                editor.putString(Constants.PREV_VM_IP, vmIp);

                Log.i(TAG, "Saving the VMM IP: " + vmmIp);
                editor.putString(Constants.PREV_VMM_IP, vmmIp);
                editor.apply();

                Log.i(TAG, "Broadcasting the details of the VM to all rapid apps");
                Intent intent = new Intent(RapidNetworkService.RAPID_VM_CHANGED);
                intent.putExtra(RapidNetworkService.RAPID_VM_IP, sClone);
                sendBroadcast(intent);

                return true;
            } else if (slamResponse == RapidMessages.ERROR) {
                Log.e(TAG, "SLAM registration replied with ERROR, VM will be null");
            } else {
                Log.e(TAG, "SLAM registration replied with uknown message " + slamResponse
                        + ", VM will be null");
            }
        } catch (IOException e) {
            Log.e(TAG, "IOException while talking to the SLAM: " + e);
        }

        return false;
    }
}
