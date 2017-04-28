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

import android.app.ProgressDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.preference.PreferenceManager;
import android.util.Log;
import android.util.SparseArray;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import javax.net.ssl.HandshakeCompletedEvent;
import javax.net.ssl.HandshakeCompletedListener;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManagerFactory;

import eu.project.rapid.ac.d2d.D2DClientService;
import eu.project.rapid.ac.d2d.PhoneSpecs;
import eu.project.rapid.ac.db.DBCache;
import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.Profiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;
import eu.project.rapid.ac.utils.Constants;
import eu.project.rapid.ac.utils.Utils;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.Configuration;
import eu.project.rapid.common.RapidConstants;
import eu.project.rapid.common.RapidConstants.COMM_TYPE;
import eu.project.rapid.common.RapidConstants.ExecLocation;
import eu.project.rapid.common.RapidMessages;
import eu.project.rapid.common.RapidMessages.AnimationMsg;
import eu.project.rapid.common.RapidUtils;
import eu.project.rapid.gvirtusfe.Frontend;

/**
 * The most important class of the framework for the client program - controls DSE, profilers,
 * communicates with remote server.
 */

public class DFE {

    // The only instance of the DFE.
    // Ignore the warning for now, also Android implements Singleton with reference to context
    // in the same way: https://developer.android.com/training/volley/requestqueue.html#singleton
    private static volatile DFE instance = null;

    private static final String TAG = "DFE";
    private static boolean useAnimationServer = false;
    private Configuration config;

    private static int mRegime;
    private static final int REGIME_CLIENT = 1;
    private static final int REGIME_SERVER = 2;
    private static COMM_TYPE commType = COMM_TYPE.SSL;
    private static boolean usePrevVm = true;
    private ExecLocation userChoice = ExecLocation.DYNAMIC;

    private long mPureRemoteDuration;
    private long prepareDataDuration;

    // private Object result;
    private String mAppName;
    private Context mContext;
    private PackageManager mPManager;

    static boolean onLineClear = false;
    static boolean onLineSSL = false;
    private static boolean isDFEActive = false;
    private int nrClones;
    private DSE mDSE;
    private NetworkProfiler netProfiler;

    // GVirtuS frontend is responsible for running the CUDA code.
    private Frontend gVirtusFrontend;
    private static Clone sClone;
    private static Socket sSocket;
    private static OutputStream sOutStream;
    private static ObjectOutputStream sObjOutStream;
    private static InputStream sInStream;
    private static ObjectInputStream sObjInStream;

    private long myId = -1;
    private String vmIp = "";
    private ArrayList<String> vmmIPs;
    private PhoneSpecs myPhoneSpecs;
    private static final int vmNrVCPUs = 1; // FIXME: number of CPUs on the VM
    private static final int vmMemSize = 512; // FIXME
    private static final int vmNrGpuCores = 1200; // FIXME

    private static ExecutorService threadPool;
    private static Set<PhoneSpecs> d2dSetPhones = new TreeSet<>();
    private static BlockingDeque<Task> tasks = new LinkedBlockingDeque<>();
    private static AtomicInteger taskId = new AtomicInteger();
    private static SparseArray<BlockingDeque<Object>> tasksResultsMap = new SparseArray<>();

    private ProgressDialog pd = null;

    /**
     * Interface to be implemented by some class that wants to be updated about some events.
     *
     * @author sokol
     */
    public interface DfeCallback {
        // Send updates about the VM connection status.

        /**
         * get formatted name based on {@link }
         **/
        void vmConnectionStatusUpdate(boolean isConnected, COMM_TYPE commType);
    }

    /**
     * Create DFE which decides where to execute remoteable methods and handles execution.
     *
     * @param clone    The clone to connect with.<br>
     *                 If null then the DFE will connect to the manager and will ask for the clone.
     * @param appName  Application name, usually the result of getPackageName().
     * @param pManager Package manager for finding apk file of the application.
     * @param context  Is the context of the application, to be passed by the activity that will create the DFE.
     */
    private DFE(String appName, PackageManager pManager, Context context, Clone clone) {

        Log.d(TAG, "DFE Created");
        isDFEActive = true;
        DFE.mRegime = REGIME_CLIENT;
        this.mAppName = appName;
        this.mPManager = pManager;
        this.mContext = context;
        sClone = clone;
        this.myPhoneSpecs = PhoneSpecs.getPhoneSpecs(mContext);
        threadPool = Executors.newFixedThreadPool(3);

        Log.i(TAG, "Current device: " + myPhoneSpecs);

        createRapidFoldersIfNotExist();
        readConfigurationFile();
        initializeCrypto();

        // The prev id is useful to the DS so that it can release already allocated VMs.
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this.mContext);
        myId = prefs.getLong(Constants.MY_OLD_ID, -1);
        vmIp = prefs.getString(Constants.PREV_VM_IP, "");
        String vmmIp = prefs.getString(Constants.PREV_VMM_IP, "");

        mDSE = DSE.getInstance(userChoice);
        netProfiler = new NetworkProfiler(mContext, config);

        start();
    }

    private void start() {
        netProfiler.registerNetworkStateTrackers();

        // Start the thread that will deal with the D2D communication
        startD2DThread();

        // Show a spinning dialog while connecting to the Manager and to the clone.
        this.pd = ProgressDialog.show(mContext, "Working...", "Initial network tasks...", true, false);
        (new InitialNetworkTasks()).execute(sClone);
    }

    /**
     * To be used on server side, only local execution
     */
    @SuppressWarnings("unused")
    private DFE() {
        mRegime = REGIME_SERVER;
    }

    /**
     * The method to be used by applications.
     *
     * @return The only instance of the DFE
     */
    public static DFE getInstance(String appName, PackageManager pManager, Context context) {
        return getInstance(appName, pManager, context, null, true, COMM_TYPE.SSL);
    }

    /**
     * The method used for testing.
     *
     * @return The only instance of the DFE
     */
    public static DFE getInstance(String appName, PackageManager pManager, Context context,
                                  Clone clone, boolean connectToPrevVm, COMM_TYPE commType) {
        // local variable increases performance by 25 percent according to
        // Joshua Bloch "Effective Java, Second Edition", p. 283-284
        DFE result = instance;

        synchronized (DFE.class) {
            if (result == null) {
                result = instance;
                if (result == null) {
                    DFE.commType = commType;
                    DFE.usePrevVm = connectToPrevVm;
                    instance = result = new DFE(appName, pManager, context, clone);
                }
            }
        }

        return result;
    }

    private void createRapidFoldersIfNotExist() {
        File rapidDir = new File(Constants.RAPID_FOLDER);
        if (!rapidDir.exists()) {
            if (!rapidDir.mkdirs()) {
                Log.w(TAG, "Could not create the Rapid folder: " + Constants.RAPID_FOLDER);
            }
        }

        File rapidTestDir = new File(Constants.TEST_LOGS_FOLDER);
        if (!rapidTestDir.exists()) {
            if (!rapidTestDir.mkdirs()) {
                Log.w(TAG, "Could not create the Rapid folder: " + Constants.TEST_LOGS_FOLDER);
            }
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

    /**
     * Create the necessary stuff for the SSL connection.
     */
    private void initializeCrypto() {
        try {
            Log.i(TAG, "Started reading the cryptographic keys");

            // Read the CA certificate
            KeyStore trustStore = KeyStore.getInstance("BKS");
            trustStore.load(new FileInputStream(Constants.SSL_CA_TRUSTSTORE), Constants.SSL_DEFAULT_PASSW.toCharArray());
            TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
            trustManagerFactory.init(trustStore);

            // Read my certificate
            KeyStore keyStore = KeyStore.getInstance("BKS");
            keyStore.load(new FileInputStream(Constants.SSL_KEYSTORE), Constants.SSL_DEFAULT_PASSW.toCharArray());
            KeyManagerFactory kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
            kmf.init(keyStore, Constants.SSL_DEFAULT_PASSW.toCharArray());

            PrivateKey myPrivateKey = (PrivateKey) keyStore.getKey(Constants.SSL_CERT_ALIAS,
                    Constants.SSL_DEFAULT_PASSW.toCharArray());
            Certificate cert = keyStore.getCertificate(Constants.SSL_CERT_ALIAS);
            PublicKey myPublicKey = cert.getPublicKey();
            Log.i(TAG, "Certificate: " + cert.toString());
            Log.i(TAG, "PrivateKey algorithm: " + myPrivateKey.getAlgorithm());
            Log.i(TAG, "PublicKey algorithm: " + myPrivateKey.getAlgorithm());

            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, trustManagerFactory.getTrustManagers(), null);
            SSLSocketFactory sslFactory = sslContext.getSocketFactory();
            Log.i(TAG, "SSL Factory created");

            config.setCryptoInitialized(true);
            config.setPrivateKey(myPrivateKey);
            config.setPublicKey(myPublicKey);
            config.setSslFactory(sslFactory);
            config.setSslContext(sslContext);

        } catch (KeyStoreException | NoSuchAlgorithmException | KeyManagementException
                | UnrecoverableKeyException | CertificateException | IOException e) {
            Log.e(TAG, "Could not initialize the crypto parameters - " + e);
        }
    }

    private class InitialNetworkTasks extends AsyncTask<Clone, String, Void> {

        private static final String TAG = "InitialNetworkTasks";

        @Override
        protected Void doInBackground(Clone... clone) {

            if (useAnimationServer) {
                // Check if the primary animation server is reachable
                boolean primaryAnimationServerReachable = RapidUtils.isPrimaryAnimationServerRunning(
                        config.getAnimationServerIp(), config.getAnimationServerPort());
                Log.i(TAG, "Primary animation server reachable: " + primaryAnimationServerReachable);
                if (!primaryAnimationServerReachable) {
                    config.setAnimationServerIp(RapidConstants.DEFAULT_SECONDARY_ANIMATION_SERVER_IP);
                    config.setAnimationServerPort(RapidConstants.DEFAULT_SECONDARY_ANIMATION_SERVER_PORT);
                }
            }

            // Anyway, show the default image, where the AC tries to connects to DS, SLAM, etc. If this
            // fails, then we will show the D2D initial image
            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_INITIAL_IMG);

            if (clone[0] == null) {
                publishProgress("Registering with the DS and the SLAM...");
                registerWithDsAndSlam();
            } else {
                publishProgress("Using the clone given by the user: " + clone[0]);
                sClone = clone[0];
            }

            config.setClone(sClone);

            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, usePrevVm ? AnimationMsg.AC_PREV_REGISTER_VM
                        : AnimationMsg.AC_NEW_REGISTER_VM);
            if (commType == COMM_TYPE.CLEAR) {
                publishProgress("Clear connection with the clone: " + sClone);
                establishConnection();
            } else { // (commType == COMM_TYPE.SSL)
                publishProgress("SSL connection with the clone: " + sClone);
                if (!establishSslConnection()) {
                    Log.w(TAG, "Setting commType to CLEAR");
                    commType = COMM_TYPE.CLEAR;
                    establishConnection();
                }
            }

            // If the connection was successful then try to send the app to the clone
            if (onLineClear || onLineSSL) {
                if (useAnimationServer) RapidUtils.sendAnimationMsg(config,
                        usePrevVm ? AnimationMsg.AC_PREV_CONN_VM : AnimationMsg.AC_NEW_CONN_VM);

                Log.i(TAG, "The communication type established with the clone is: " + commType);

                if (config.getGvirtusIp() == null) {
                    // If gvirtusIp is null, then gvirtus backend is running on the physical machine where
                    // the VM is running.
                    // Try to find a way here to get the ip address of the physical machine.
                    // config.setGvirtusIp(TODO: ip address of the physical machine where the VM is running);
                }

                publishProgress("Registering application with the RAPID system...");
                if (useAnimationServer) RapidUtils.sendAnimationMsg(config,
                        usePrevVm ? AnimationMsg.AC_PREV_APK_VM : AnimationMsg.AC_NEW_APK_VM);
                sendApk(sInStream, sOutStream, sObjOutStream);

                // Find rtt to the server
                // Measure the data rate when just connected
                publishProgress("Sending/receiving data for 3 seconds to measure the ulRate and dlRate...");
                if (useAnimationServer) RapidUtils.sendAnimationMsg(config,
                        usePrevVm ? AnimationMsg.AC_PREV_RTT_VM : AnimationMsg.AC_NEW_RTT_VM);
                NetworkProfiler.rttPing(sInStream, sOutStream);
                if (useAnimationServer)
                    RapidUtils.sendAnimationMsg(config, usePrevVm ? AnimationMsg.AC_PREV_DL_RATE_VM
                            : AnimationMsg.AC_NEW_DL_RATE_VM);
                NetworkProfiler.measureDlRate(sClone.getIp(), config.getClonePortBandwidthTest());
                if (useAnimationServer)
                    RapidUtils.sendAnimationMsg(config, usePrevVm ? AnimationMsg.AC_PREV_UL_RATE_VM
                            : AnimationMsg.AC_NEW_UL_RATE_VM);
                NetworkProfiler.measureUlRate(sClone.getIp(), config.getClonePortBandwidthTest());
                if (useAnimationServer) RapidUtils.sendAnimationMsg(config, usePrevVm
                        ? AnimationMsg.AC_PREV_REGISTRATION_OK_VM : AnimationMsg.AC_NEW_REGISTRATION_OK_VM);

                try {
                    ((DfeCallback) mContext).vmConnectionStatusUpdate(onLineClear || onLineSSL, commType);
                } catch (ClassCastException e) {
                    Log.i(TAG, "This class doesn't implement callback methods.");
                }

                // Start 10 TaskRunner threads that will handle the task dispatching process.
                for (int i = 0; i < 10; i++) {
                    threadPool.submit(new TaskRunner(i));
                }

            } else {
                if (useAnimationServer)
                    RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_REGISTER_VM_ERROR);
            }

            // if (config.getGvirtusIp() != null) {
            // // Create a gvirtus frontend object that is responsible for executing the CUDA code.
            // publishProgress("Connecting with GVirtuS backend...");
            // gVirtusFrontend = new Frontend(config.getGvirtusIp(), config.getGvirtusPort());
            // }

            return null;
        }

        @Override
        protected void onProgressUpdate(String... progress) {
            Log.i(TAG, progress[0]);
            if (pd != null) {
                pd.setMessage(progress[0]);
            }
        }

        @Override
        protected void onPostExecute(Void result) {
            Log.i(TAG, "Finished initial network tasks");
            if (pd != null) {
                pd.dismiss();
            }
        }

        @Override
        protected void onPreExecute() {
            Log.i(TAG, "Started initial network tasks");
        }

        private boolean registerWithDsAndSlam() {
            Log.i(TAG, "Registering...");
            boolean registeredWithSlam = false;

            if (registerWithDs()) {
                // register with SLAM
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
            if (useAnimationServer) RapidUtils.sendAnimationMsg(config,
                    usePrevVm ? AnimationMsg.AC_PREV_VM_DS : AnimationMsg.AC_NEW_REGISTER_DS);

            Socket dsSocket = null;
            ObjectOutputStream dsOut = null;
            ObjectInputStream dsIn = null;

            Log.i(TAG, "Registering with DS " + config.getDSIp() + ":" + config.getDSPort());
            try {
                dsSocket = new Socket();
                dsSocket.connect(new InetSocketAddress(config.getDSIp(), config.getDSPort()), 10 * 1000);
                dsOut = new ObjectOutputStream(dsSocket.getOutputStream());
                dsIn = new ObjectInputStream(dsSocket.getInputStream());

                // Send the name and id to the DS
                if (usePrevVm) {
                    if (useAnimationServer)
                        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_PREV_VM_DS);
                    Log.i(TAG, "AC_REGISTER_PREV_DS");
                    // Send message format: command (java byte), userId (java long), qosFlag (java int)
                    dsOut.writeByte(RapidMessages.AC_REGISTER_PREV_DS);
                    dsOut.writeLong(myId); // send my user ID so that my previous VM can be released
                } else { // Connect to a new VM
                    if (useAnimationServer)
                        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_NEW_REGISTER_DS);
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
            } finally {
                RapidUtils.closeQuietly(dsOut);
                RapidUtils.closeQuietly(dsIn);
                RapidUtils.closeQuietly(dsSocket);
            }

            return false;
        }

        /**
         * Register with the SLAM.
         */
        private boolean registerWithSlam(String vmmIp) {

            Socket slamSocket = null;
            ObjectOutputStream oos = null;
            ObjectInputStream ois = null;

            Log.i(TAG, "Registering with SLAM " + config.getSlamIp() + ":" + config.getSlamPort());
            try {

                slamSocket = new Socket();
                slamSocket.connect(new InetSocketAddress(config.getSlamIp(), config.getSlamPort()),
                        10 * 1000);

                oos = new ObjectOutputStream(slamSocket.getOutputStream());
                ois = new ObjectInputStream(slamSocket.getInputStream());

                if (useAnimationServer) RapidUtils.sendAnimationMsg(config,
                        usePrevVm ? AnimationMsg.AC_PREV_REGISTER_SLAM : AnimationMsg.AC_NEW_REGISTER_SLAM);

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
                    SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(mContext);
                    SharedPreferences.Editor editor = prefs.edit();

                    editor.putLong(Constants.MY_OLD_ID, myId);
                    editor.putString(Constants.PREV_VM_IP, vmIp);

                    Log.i(TAG, "Saving the VMM IP: " + vmmIp);
                    editor.putString(Constants.PREV_VMM_IP, vmmIp);
                    editor.apply();

                    return true;
                } else if (slamResponse == RapidMessages.ERROR) {
                    Log.e(TAG, "SLAM registration replied with ERROR, VM will be null");
                } else {
                    Log.e(TAG, "SLAM registration replied with uknown message " + slamResponse
                            + ", VM will be null");
                }
            } catch (IOException e) {
                Log.e(TAG, "IOException while talking to the SLAM: " + e);
            } finally {
                RapidUtils.closeQuietly(oos);
                RapidUtils.closeQuietly(ois);
                RapidUtils.closeQuietly(slamSocket);
            }

            return false;
        }
    }

    public void onDestroy() {
        Log.d(TAG, "onDestroy");
        isDFEActive = false;
        DeviceProfiler.onDestroy();
        netProfiler.onDestroy();
        DBCache.saveDbCache();
        closeConnection();
        threadPool.shutdownNow();
        instance = null;
        // FIXME Should I also stop the D2D listening service here or should I leave it running?
        if (useAnimationServer) RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_INITIAL_IMG);
    }

    private void startD2DThread() {
        Log.i(TAG, "Starting the D2D listening service...");
        Intent d2dServiceIntent = new Intent(mContext, D2DClientService.class);
        mContext.startService(d2dServiceIntent);

        // This thread will run with a certain frequency to read the D2D devices that are written by the
        // D2DClientService on the sdcard. This is needed because another DFE may have written the
        // devices on the file.
        ScheduledThreadPoolExecutor d2dSetReaderThread = (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(5);
        d2dSetReaderThread.scheduleWithFixedDelay(new D2DSetReader(), 0,
                D2DClientService.FREQUENCY_READ_D2D_SET, TimeUnit.MILLISECONDS);
    }

    private class D2DSetReader implements Runnable {

        @SuppressWarnings("unchecked")
        @Override
        public void run() {
            try {
                Log.i(TAG, "Reading the saved D2D phones...");
                d2dSetPhones = (Set<PhoneSpecs>) Utils.readObjectFromFile(Constants.FILE_D2D_PHONES);

                if (d2dSetPhones != null && d2dSetPhones.size() > 0) {
                    if (useAnimationServer)
                        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_RECEIVED_D2D);
                } else {
                    if (useAnimationServer)
                        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_NO_MORE_D2D);
                }

                Log.i(TAG, "List of D2D phones:");
                for (PhoneSpecs p : d2dSetPhones) {
                    Log.i(TAG, p.toString());
                }
            } catch (IOException | ClassNotFoundException e) {
                Log.e(TAG, "Error on D2DSetReader while trying to read the saved set of D2D phones: " + e);
            }
        }
    }

    /**
     * Set up streams for the socket connection, perform initial communication with the clone.
     */
    private boolean establishConnection() {
        try {
            long sTime = System.nanoTime();
            long startTxBytes = NetworkProfiler.getProcessTxBytes();
            long startRxBytes = NetworkProfiler.getProcessRxBytes();

            Log.i(TAG, "Connecting in CLEAR with AS on: " + sClone.getIp() + ":" + sClone.getPort());
            sSocket = new Socket();
            sSocket.connect(new InetSocketAddress(sClone.getIp(), sClone.getPort()), 10 * 1000);

            sOutStream = sSocket.getOutputStream();
            sInStream = sSocket.getInputStream();
            sObjOutStream = new ObjectOutputStream(sOutStream);
            sObjInStream = new ObjectInputStream(sInStream);

            long dur = System.nanoTime() - sTime;
            long totalTxBytes = NetworkProfiler.getProcessTxBytes() - startTxBytes;
            long totalRxBytes = NetworkProfiler.getProcessRxBytes() - startRxBytes;

            Log.d(TAG, "Socket and streams set-up time - " + dur / 1000000 + "ms");
            Log.d(TAG, "Total bytes sent: " + totalTxBytes);
            Log.d(TAG, "Total bytes received: " + totalRxBytes);
            return onLineClear = true;

        } catch (Exception e) {
            e.printStackTrace();
            fallBackToLocalExecution("Connection setup with the clone failed: " + e);
        } finally {
            onLineSSL = false;
        }
        return onLineClear = false;
    }

    /**
     * Set up streams for the secure socket connection, perform initial communication with the server.
     */
    private boolean establishSslConnection() {

        if (!config.isCryptoInitialized()) {
            Log.e(TAG,
                    "The initialization of the cryptographic keys is not done correctly. Cannot perform ssl connection!");
            return false;
        }

        try {
            // if (useAnimationServer) RapidUtils.sendAnimationMsg(config, RapidMessages.AC_CONNECT_VM);

            Long sTime = System.nanoTime();
            long startTxBytes = NetworkProfiler.getProcessTxBytes();
            long startRxBytes = NetworkProfiler.getProcessRxBytes();

            Log.i(TAG, "Connecting in SSL with clone: " + sClone.getIp() + ":" + sClone.getSslPort());

            sSocket = config.getSslFactory().createSocket(sClone.getIp(), sClone.getSslPort());
            // Log.i(TAG, "getEnableSessionCreation: " + ((SSLSocket)
            // sSocket).getEnableSessionCreation());
            // ((SSLSocket) sSocket).setEnableSessionCreation(false);

            // sslContext.getClientSessionContext().getSession(null).invalidate();

            ((SSLSocket) sSocket).addHandshakeCompletedListener(new SSLHandshakeCompletedListener());
            Log.i(TAG, "socket created");

            // Log.i(TAG, "Enabled cipher suites: ");
            // for (String s : ((SSLSocket) sSocket).getEnabledCipherSuites()) {
            // Log.i(TAG, s);
            // }

            sOutStream = sSocket.getOutputStream();
            sInStream = sSocket.getInputStream();
            sObjOutStream = new ObjectOutputStream(sOutStream);
            sObjInStream = new ObjectInputStream(sInStream);

            long dur = System.nanoTime() - sTime;
            long totalTxBytes = NetworkProfiler.getProcessTxBytes() - startTxBytes;
            long totalRxBytes = NetworkProfiler.getProcessRxBytes() - startRxBytes;

            Log.d(TAG, "Socket and streams set-up time - " + dur / 1000000 + "ms");
            Log.d(TAG, "Total bytes sent: " + totalTxBytes);
            Log.d(TAG, "Total bytes received: " + totalRxBytes);
            return onLineSSL = true;

        } catch (UnknownHostException e) {
            e.printStackTrace();
            fallBackToLocalExecution("UnknownHostException - Connection setup to server failed: " + e);
        } catch (IOException e) {
            e.printStackTrace();
            fallBackToLocalExecution("IOException - Connection setup to server failed: " + e);
        } catch (Exception e) {
            e.printStackTrace();
            fallBackToLocalExecution("Exception - Connection setup to server failed: " + e);
        } finally {
            onLineClear = false;
        }

        return onLineSSL = false;
    }

    private class SSLHandshakeCompletedListener implements HandshakeCompletedListener {

        @Override
        public void handshakeCompleted(HandshakeCompletedEvent event) {

            try {
                Log.i(TAG, "SSL handshake completed");
                // Log.i(TAG, "getCipherSuite: " + event.getCipherSuite());
                // Log.i(TAG, "algorithm: " + config.getPrivateKey().getAlgorithm());
                // Log.i(TAG, "modulusBitLength: " + ((RSAPrivateKey)
                // config.getPrivateKey()).getModulus().bitLength());

                // SSLSession session = event.getSession();
                // Log.i(TAG, "getProtocol: " + session.getProtocol());
                // Log.i(TAG, "getPeerHost: " + session.getPeerHost());
                // Log.i(TAG, "getId: " + RapidUtils.bytesToHex(session.getId()));
                // Log.i(TAG, "getCreationTime: " + session.getCreationTime());

                // java.security.cert.Certificate[] certs = event.getPeerCertificates();
                // for (int i = 0; i < certs.length; i++)
                // {
                // if (!(certs[i] instanceof java.security.cert.X509Certificate)) continue;
                // java.security.cert.X509Certificate cert = (java.security.cert.X509Certificate) certs[i];
                // Log.i(TAG, "Cert #" + i + ": " + cert.getSubjectDN().getName());
                // }
            } catch (Exception e) {
                Log.e(TAG, "SSL handshake completed with errors: " + e);
            }
        }

    }

    private void closeConnection() {
        // if (useAnimationServer) RapidUtils.sendAnimationMsg(config, RapidMessages.AC_DISCONNECT_VM);
        new Thread(
                new Runnable() {
                    @Override
                    public void run() {
                        RapidUtils.closeQuietly(sObjOutStream);
                        RapidUtils.closeQuietly(sObjInStream);
                        RapidUtils.closeQuietly(sSocket);
                        onLineClear = onLineSSL = false;
                    }
                }
        ).start();
    }

    /**
     * Send APK file to the remote server
     */
    private void sendApk(InputStream is, OutputStream os, ObjectOutputStream oos) {

        try {
            Log.d(TAG, "Getting apk data");
            String apkName = mPManager.getApplicationInfo(mAppName, 0).sourceDir;
            File apkFile = new File(apkName);
            Log.d(TAG, "Apk name - " + apkName);

            os.write(RapidMessages.AC_REGISTER_AS);
            // Send apkName and apkLength to clone.
            // The clone will compare these information with what he has and tell
            // if he doesn't have the apk or this one differs in size.
            oos.writeObject(mAppName);
            oos.writeInt((int) apkFile.length());
            oos.flush();
            int response = is.read();

            if (response == RapidMessages.AS_APP_REQ_AC) {
                // Send the APK file if needed

                FileInputStream fin = new FileInputStream(apkFile);
                BufferedInputStream bis = new BufferedInputStream(fin);

                // Send the file
                Log.d(TAG, "Sending apk");
                int BUFFER_SIZE = 8192;
                byte[] tempArray = new byte[BUFFER_SIZE];
                int read;
                while ((read = bis.read(tempArray, 0, tempArray.length)) > -1) {
                    oos.write(tempArray, 0, read);
                    // Log.d(TAG, "Sent " + totalRead + " of " + apkFile.length() + " bytes");
                }
                oos.flush();
                RapidUtils.closeQuietly(bis);
            }
        } catch (IOException e) {
            fallBackToLocalExecution("IOException: " + e.getMessage());
        } catch (NameNotFoundException e) {
            fallBackToLocalExecution("Application not found: " + e.getMessage());
        } catch (Exception e) {
            fallBackToLocalExecution("Exception: " + e.getMessage());
        }
    }

    private void fallBackToLocalExecution(String message) {
        Log.e(TAG, message);
        onLineClear = onLineSSL = false;
    }

    /**
     * @param appName    The application name.
     * @param methodName The current method that we want to offload from this application.<br>
     *                   Different methods of the same application will have a different set of parameters.
     * @return The execution location which can be one of: LOCAL, REMOTE.<br>
     */
    private ExecLocation findExecLocation(String appName, String methodName) {

        ExecLocation execLocation = mDSE.findExecLocation(appName, methodName);
        Log.i(TAG, "Execution location: " + execLocation);

        return execLocation;
    }

    /**
     * Ask the DSE to find the best offloading scheme based on user choice. Developer can use this
     * method in order to prepare the data input based on the decision the framework will make about
     * the execution location.
     *
     * @param methodName The name of the method that will be executed.
     * @return The execution location which can be one of: LOCAL, REMOTE.<br>
     */
    private ExecLocation findExecLocation(String methodName) {
        return findExecLocation(mAppName, methodName);
    }

    /**
     * Wrapper of the execute method with no parameters for the executable method
     *
     * @param m The method to be executed.
     * @param o The object calling the method.
     * @return The result of the method execution, which can also be an exception if something went wrong.
     * @throws Throwable
     */
    public Object execute(Method m, Object o) throws Throwable {
        return execute(m, null, o);
    }

    /**
     * Call DSE to decide where to execute the operation, start profilers, execute (either locally or
     * remotely), collect profiling data and return execution results.
     *
     * @param m       method to be executed
     * @param pValues with parameter values
     * @param o       on object
     * @return result of execution, or an exception if it happened
     * @throws NoSuchMethodException    If the method was not found in the object class.
     * @throws ClassNotFoundException   If the class of the object could not be found by the classloader.
     * @throws IllegalAccessException
     * @throws SecurityException
     * @throws IllegalArgumentException If the arguments passed to the method are not correct.
     */
    public Object execute(Method m, Object[] pValues, Object o) throws IllegalArgumentException,
            SecurityException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {

        Object result = null;
        try {
            int id = taskId.incrementAndGet();
            tasksResultsMap.put(id, new LinkedBlockingDeque<>());
            Log.v(TAG, "Adding task on the tasks blocking queue...");
            tasks.put(new Task(id, m, pValues, o));
            Log.v(TAG, "Task added");

            Log.v(TAG, "Waiting for the result of this task to be inserted in the queue by the working thread...");
            result = tasksResultsMap.get(id).take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return result;
    }

    private class Task {
        Method m;
        Object[] pValues;
        Object o;
        int id;

        Task(int id, Method m, Object[] pValues, Object o) {
            this.id = id;
            this.m = m;
            this.pValues = pValues;
            this.o = o;
        }
    }

    private class TaskRunner implements Runnable {
        private final String TAG;

        public TaskRunner(int id) {
            TAG = "DFE-TaskRunner-" + id;
        }

        @Override
        public void run() {

            try (Socket s = new Socket(sClone.getIp(), sClone.getPort());
                 OutputStream os = s.getOutputStream();
                 InputStream is = s.getInputStream();
                 ObjectOutputStream oos = new ObjectOutputStream(os);
                 ObjectInputStream ois = new ObjectInputStream(is)) {

                // I know that the DFE main thread has already sent the APK.
                // However, I need to tell the listening threads which application is being connected,
                // so that they can load the classes and libraries.
                sendApk(is, os, oos);

                while (true) {
                    try {
                        Log.v(TAG, "Waiting for task...");
                        Task task = tasks.take();
                        Log.v(TAG, "Got a task, executing...");
                        Object result = runTask(task, os, ois, oos);
                        Log.v(TAG, "Task finished execution, putting result on the resultMap...");
                        tasksResultsMap.get(task.id).put(result);
                        Log.v(TAG, "Result inserted on the resultMap.");
                    } catch (InterruptedException e) {
                        if (!isDFEActive) {
                            Log.v(TAG, "The DFE is destroyed, exiting");
                            break;
                        } else {
                            Thread.currentThread().interrupt();
                        }
                    }
                }

            } catch (UnknownHostException e) {
                Log.e(TAG, "UnknownHostException while connecting: " + e);
            } catch (IOException e) {
                Log.e(TAG, "IOException while connecting: " + e);
            }
        }

        private Object runTask(Task task, OutputStream os, ObjectInputStream ois, ObjectOutputStream oos) {

            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_INITIAL_IMG);
            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_PREPARE_DATA);

            Object result = null;
            ExecLocation execLocation = findExecLocation(task.m.getName());
            if (execLocation.equals(ExecLocation.LOCAL)) {
                Log.v(TAG, "Should run the method locally...");
                // First try to see if we can offload this task to a more powerful device that is in D2D
                // distance.
                // Do this only if we are not connected to a clone, otherwise it becomes a mess.
                if (!onLineClear && !onLineSSL) {
                    if (d2dSetPhones != null && d2dSetPhones.size() > 0) {
                        Log.v(TAG, "Running the method using D2D...");
                        Iterator<PhoneSpecs> it = d2dSetPhones.iterator();
                        // This is the best phone from the D2D ones since the set is sorted and this is the
                        // first element.
                        PhoneSpecs otherPhone = it.next();
                        if (otherPhone.compareTo(myPhoneSpecs) > 0) {
                            result = executeD2D(task, otherPhone);
                        }
                    }
                }

                // If the D2D execution didn't take place or something happened that the execution was
                // interrupted the result would still be null.
                if (result == null) {
                    Log.v(TAG, "No D2D execution was performed, running locally...");
                    if (useAnimationServer)
                        RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_DECISION_LOCAL);
                    result = executeLocally(task);
                }
            } else if (execLocation.equals(ExecLocation.REMOTE)) {
                result = executeRemotely(task, os, ois, oos);
                if (result instanceof InvocationTargetException) {
                    // The remote execution throwed an exception, try to run the method locally.
                    Log.w(TAG, "The result was InvocationTargetException. Running the method locally...");
                    result = executeLocally(task);
                }
            }

            return result;
        }

        /**
         * Execute the method locally
         *
         * @return result of execution, or an exception if it happened
         * @throws IllegalAccessException
         * @throws SecurityException
         * @throws IllegalArgumentException If the arguments passed to the method are not correct.
         */
        private Object executeLocally(Task task) {

            ProgramProfiler progProfiler = new ProgramProfiler(mAppName, task.m.getName());
            DeviceProfiler devProfiler = new DeviceProfiler(mContext);
            Profiler profiler = new Profiler(mRegime, progProfiler, null, devProfiler);

            // Start tracking execution statistics for the method
            profiler.startExecutionInfoTracking();
            Object result = null; // Access it
            try {
                // Make sure that the method is accessible
                // if (useAnimationServer) RapidUtils.sendAnimationMsg(config, RapidMessages.AC_EXEC_LOCAL);
                Long startTime = System.nanoTime();
                task.m.setAccessible(true);

                result = task.m.invoke(task.o, task.pValues);
                long mPureLocalDuration = System.nanoTime() - startTime;
                Log.d(TAG, "LOCAL " + task.m.getName() + ": Actual Invocation duration - "
                        + mPureLocalDuration / 1000000 + "ms");

                // Collect execution statistics
                profiler.stopAndLogExecutionInfoTracking(prepareDataDuration, mPureLocalDuration);
            } catch (IllegalAccessException | InvocationTargetException | IllegalArgumentException e) {
                Log.w(TAG, "Exception while running the method locally: " + e);
                profiler.stopAndDiscardExecutionInfoTracking();
                e.printStackTrace();
            }

            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_LOCAL_FINISHED);

            return result;
        }

        /**
         * Execute the method on a phone that is close to us so that we can connect on D2D mode. If we
         * are here, it means that this phone is not connected to a clone, so we can define the clone to
         * be this D2D device.
         *
         * @param otherPhone Is the closeby phones, which is willing to help with offloading.
         * @return
         * @throws IllegalArgumentException
         * @throws SecurityException
         */
        private Object executeD2D(Task task, PhoneSpecs otherPhone) {

            // otherPhone.setIp("192.168.43.1");
            Log.i(TAG, "Trying to execute the method using D2D on device: " + otherPhone);
            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_OFFLOAD_D2D);

            Object result = null;
            try (Socket socket = new Socket(otherPhone.getIp(), config.getClonePort());
                 InputStream is = socket.getInputStream();
                 OutputStream os = socket.getOutputStream();
                 ObjectInputStream ois = new ObjectInputStream(is);
                 ObjectOutputStream oos = new ObjectOutputStream(os)) {

                sendApk(is, os, oos);
                result = executeRemotely(task, os, ois, oos);
            } catch (Exception e) {
                Log.w(TAG, "Exception while trying to connect with other phone: " + e);
            }
            return result;
        }

        /**
         * Execute method remotely
         *
         * @return result of execution, or an exception if it happened
         * @throws NoSuchMethodException    If the method was not found in the object class.
         * @throws ClassNotFoundException   If the class of the object could not be found by the classloader.
         * @throws IllegalAccessException
         * @throws SecurityException
         * @throws IllegalArgumentException If the arguments passed to the method are not correct.
         */
        private Object executeRemotely(Task task, OutputStream os, ObjectInputStream ois,
                                       ObjectOutputStream oos) {


            if (useAnimationServer)
                RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_DECISION_OFFLOAD_AS);

            // Maybe the developer has implemented the prepareDataOnClient() method that helps him prepare
            // the data based on where the execution will take place then call it.
            // Prepare the data by calling the prepareData(localFraction) implemented by the developer.
            try {
                long s = System.nanoTime();
                Method prepareDataMethod = task.o.getClass().getDeclaredMethod("prepareDataOnClient");
                prepareDataMethod.setAccessible(true);
                prepareDataMethod.invoke(task.o);
                prepareDataDuration = System.nanoTime() - s;
            } catch (NoSuchMethodException e) {
                Log.w(TAG, "The method prepareDataOnClient() does not exist");
            } catch (InvocationTargetException | IllegalAccessException | IllegalArgumentException e) {
                Log.e(TAG, "Exception while calling method prepareDataOnClient(): " + e);
                e.printStackTrace();
            }

            ProgramProfiler progProfiler = new ProgramProfiler(mAppName, task.m.getName());
            DeviceProfiler devProfiler = new DeviceProfiler(mContext);
            NetworkProfiler netProfiler = new NetworkProfiler();
            Profiler profiler = new Profiler(mRegime, progProfiler, netProfiler, devProfiler);

            // Start tracking execution statistics for the method
            profiler.startExecutionInfoTracking();

            Object result;
            try {
                Long startTime = System.nanoTime();
                os.write(RapidMessages.AC_OFFLOAD_REQ_AS);
                result = sendAndExecute(task, ois, oos);

                Long duration = System.nanoTime() - startTime;
                Log.d(TAG, "REMOTE " + task.m.getName() + ": Actual Send-Receive duration - "
                        + duration / 1000000 + "ms");
                // Collect execution statistics
                profiler.stopAndLogExecutionInfoTracking(prepareDataDuration, mPureRemoteDuration);

                if (useAnimationServer)
                    RapidUtils.sendAnimationMsg(config, AnimationMsg.AC_OFFLOADING_FINISHED);
            } catch (Exception e) {
                // No such host exists, execute locally
                Log.e(TAG, "REMOTE ERROR: " + task.m.getName() + ": " + e);
                e.printStackTrace();
                profiler.stopAndDiscardExecutionInfoTracking();
                result = executeLocally(task);
                // ConnectionRepair repair = new ConnectionRepair();
                // repair.start();
            }
            return result;
        }

        /**
         * Send the object, the method to be executed and parameter values to the remote server for
         * execution.
         *
         * @return result of the remoted method or an exception that occurs during execution
         * @throws IOException
         * @throws ClassNotFoundException
         * @throws NoSuchMethodException
         * @throws InvocationTargetException
         * @throws IllegalAccessException
         * @throws SecurityException
         * @throws IllegalArgumentException
         */
        private Object sendAndExecute(Task task, ObjectInputStream ois, ObjectOutputStream oos)
                throws IOException, ClassNotFoundException, IllegalArgumentException, SecurityException,
                IllegalAccessException, InvocationTargetException, NoSuchMethodException {

            // Send the object itself
            sendObject(task, oos);

            // TODO To be more precise, this message should be sent by the AS, but for simplicity I put it
            // here.
            if (useAnimationServer) RapidUtils.sendAnimationMsg(config, AnimationMsg.AS_RUN_METHOD);

            // Read the results from the server
            Log.d(TAG, "Read Result");
            Object response = ois.readObject();

            // TODO To be more precise, this message should be sent by the AS, but for simplicity I put it
            // here.
            if (useAnimationServer) RapidUtils.sendAnimationMsg(config, AnimationMsg.AS_RESULT_AC);

            ResultContainer container = (ResultContainer) response;
            Object result;

            Class<?>[] pTypes = {Remoteable.class};
            try {
                // Use the copyState method that must be defined for all Remoteable
                // classes to copy the state of relevant fields to the local object
                task.o.getClass().getMethod("copyState", pTypes).invoke(task.o, container.objState);
            } catch (NullPointerException e) {
                // Do nothing - exception happened remotely and hence there is
                // no object state returned.
                // The exception will be returned in the function result anyway.
                Log.d(TAG, "Exception received from remote server - " + container.functionResult);
            }

            result = container.functionResult;
            mPureRemoteDuration = container.pureExecutionDuration;

            Log.d(TAG, "Finished remote execution");

            return result;
        }
    }

    /**
     * Send the object (along with method and parameters) to the remote server for execution
     *
     * @throws IOException
     */
    private void sendObject(Task task, ObjectOutputStream oos)
            throws IOException {
        oos.reset();
        Log.d(TAG, "Write Object and data");

        // Send the number of clones needed to execute the method
        oos.writeInt(nrClones);

        // Send object for execution
        oos.writeObject(task.o);

        // Send the method to be executed
        // Log.d(TAG, "Write Method - " + m.getName());
        oos.writeObject(task.m.getName());

        // Log.d(TAG, "Write method parameter types");
        oos.writeObject(task.m.getParameterTypes());

        // Log.d(TAG, "Write method parameter values");
        oos.writeObject(task.pValues);
        oos.flush();
    }

    /**
     * Take care of a broken connection - try restarting it when something breaks down immediately or
     * alternatively listen to changing network conditions
     */
    public class ConnectionRepair extends Thread {
        /**
         * Simple reestablish the connection
         */
        @Override
        public void run() {
            // Try simply restarting the connection
            Log.d(TAG, "Trying to reestablish connection to the server");
            synchronized (this) {
                if (!onLineClear && !onLineSSL) {
                    // Show a spinning dialog while connecting to the Manager and to the clone.
                    // pd = ProgressDialog.show(mContext, "Working..", "Initial network tasks...", true,
                    // false);
                    (new InitialNetworkTasks()).execute(sClone);

                    // establishConnection();
                }
            }

            // If still offline, establish intent listeners that would try to
            // restart the connection when the service comes back up
            synchronized (this) {
                if (!onLineClear && !onLineSSL) {
                    Log.d(TAG, "Reestablishing failed - register listeners for reconnecting");
                    final ConnectivityManager connectivityManager =
                            (ConnectivityManager) mContext.getSystemService(Context.CONNECTIVITY_SERVICE);

                    BroadcastReceiver networkStateReceiver = new BroadcastReceiver() {
                        public void onReceive(Context context, Intent intent) {
                            context.unregisterReceiver(this);
                            NetworkInfo netInfo = connectivityManager.getActiveNetworkInfo();
                            if (netInfo != null) {
                                Log.d(TAG, "Network back up, try reestablishing the connection");
                                // establishConnection();

                                pd = ProgressDialog.show(mContext, "Working..", "Initial network tasks...", true,
                                        false);
                                (new InitialNetworkTasks()).execute(sClone);
                            }
                        }
                    };
                    IntentFilter networkStateFilter =
                            new IntentFilter(ConnectivityManager.CONNECTIVITY_ACTION);
                    mContext.registerReceiver(networkStateReceiver, networkStateFilter);
                }
            }
        }
    }

    @SuppressWarnings("unused")
    public String getConnectionType() {
        return NetworkProfiler.currentNetworkTypeName;
    }

    public void setUserChoice(ExecLocation userChoice) {
        this.userChoice = userChoice;
        mDSE.setUserChoice(userChoice);
    }

    @SuppressWarnings("unused")
    public ExecLocation getUserChoice() {
        return userChoice;
    }

    @SuppressWarnings("unused")
    public int getRegime() {
        return mRegime;
    }

    public void setNrClones(int nrClones) {
        Log.i(TAG, "Changing nrClones to: " + nrClones);
        this.nrClones = nrClones;
    }

    public Frontend getGvirtusFrontend() {
        return gVirtusFrontend;
    }

    @SuppressWarnings("unused")
    public void setGvirtusFrontend(Frontend gVirtusFrontend) {
        this.gVirtusFrontend = gVirtusFrontend;
    }

    @SuppressWarnings("unused")
    public Configuration getConfig() {
        return config;
    }

    public Context getContext() {
        return mContext;
    }

    public ExecLocation getLastExecLocation(String appName, String methodName) {
        return mDSE.getLastExecLocation(appName, methodName);
    }

    public long getLastExecDuration(String appName, String methodName) {
        return mDSE.getLastExecDuration(appName, methodName);
    }

    public static boolean isUseAnimationServer() {
        return useAnimationServer;
    }

    public static void setUseAnimationServer(boolean useAnimationServer) {
        DFE.useAnimationServer = useAnimationServer;
    }

    @SuppressWarnings("unused")
    private void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
