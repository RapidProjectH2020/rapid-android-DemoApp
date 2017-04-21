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

import android.util.Log;

import java.util.Deque;
import java.util.Iterator;

import eu.project.rapid.ac.db.DBCache;
import eu.project.rapid.ac.db.DBEntry;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.common.RapidConstants.ExecLocation;

/**
 * DSE decides whether to execute the requested method locally or remotely.
 */
class DSE {

    private static DSE instance;

    private static final String TAG = "DSE";
    private static final boolean VERBOSE_LOG = false;

    // To be used in case of no previous remote execution.
    // If the ulRate and dlRate are bigger than these values then we offload
    private static final int MIN_UL_RATE_OFFLOAD_1_TIME = 256 * 1000; // b/s
    private static final int MIN_DL_RATE_OFFLOAD_1_TIME = 256 * 1000; // b/s

    private ExecLocation userChoice;
    private DBCache dbCache;

    private DSE(ExecLocation userChoice) {
        this.userChoice = userChoice;
        this.dbCache = DBCache.getDbCache();
    }

    public static DSE getInstance(ExecLocation userChoice) {
        // local variable increases performance by 25 percent according to
        // Joshua Bloch "Effective Java, Second Edition", p. 283-284
        DSE result = instance;

        if (result == null) {
            synchronized (DFE.class) {
                result = instance;
                if (result == null) {
                    instance = result = new DSE(userChoice);
                }
            }
        }

        return result;
    }

    /**
     * Decide whether to execute remotely, locally, or hybrid
     *
     * @return The type of execution: LOCAL, REMOTE, HYBRID
     */
    ExecLocation findExecLocation(String appName, String methodName) {
        Log.v(TAG, "execLoc: " + DFE.onLineClear + ", " + DFE.onLineSSL + ", " + userChoice);
        if (!DFE.onLineClear && !DFE.onLineSSL || userChoice.equals(ExecLocation.LOCAL)) {
            return ExecLocation.LOCAL;
        } else if (userChoice.equals(ExecLocation.REMOTE)) {
            return ExecLocation.REMOTE;
        } else { // if (userChoice.equals(ExecLocation.DYNAMIC)) {
            int ulRate = NetworkProfiler.lastUlRate.getBw();
            int dlRate = NetworkProfiler.lastDlRate.getBw();

            if (shouldOffloadDBCache(appName, methodName, ulRate, dlRate)) {
                Log.d(TAG, "Execute Remotely - True");
                return ExecLocation.REMOTE;
            } else {
                Log.d(TAG, "Execute Remotely - False");
                return ExecLocation.LOCAL;
            }
        }
    }

    /**
     * @param appName    The application name.
     * @param methodName The method name.
     * @param currUlRate The current upload rate.
     * @param currDlRate The current download rate.
     * @return <b>True</b> if the method should be executed remotely<br>
     * <b>False</b> otherwise.
     */
    private boolean shouldOffloadDBCache(String appName, String methodName, int currUlRate,
                                         int currDlRate) {

        Log.i(TAG,
                "Trying to decide using DB cache where to execute the method: appName=" + appName
                        + ", methodName=" + methodName + ", currUlRate=" + currUlRate + ", currDlRate="
                        + currDlRate);
        Log.i(TAG, String.format("DB cache has %d entries and %d measurements", dbCache.size(),
                dbCache.nrElements()));

        // Variables needed for the local executions
        int nrLocalExec;
        double meanDurLocal = 0, meanEnergyLocal = 0;

        // Variables needed for the remote executions
        int nrRemoteExec;
        double meanDurRemote1 = 0, meanEnergyRemote1 = 0;
        double meanDurRemote2, meanEnergyRemote2;
        double meanDurRemote, meanEnergyRemote;

        // Check if the method has been executed LOCALLY in previous runs
        Deque<DBEntry> localResults = dbCache.getAllEntriesFilteredOn(appName, methodName, ExecLocation.LOCAL);
        nrLocalExec = localResults.size();

        // Check if the method has been executed REMOTELY in previous runs
        Deque<DBEntry> remoteResults = dbCache.getAllEntriesFilteredOn(appName, methodName, ExecLocation.REMOTE,
                NetworkProfiler.currentNetworkTypeName, NetworkProfiler.currentNetworkSubtypeName);
        nrRemoteExec = remoteResults.size();

        // DECISION 1
        // If the number of previous remote executions is zero and the current connection is good
        // then offload the method to see how it goes.
        if (nrRemoteExec == 0) {
            if (currUlRate > MIN_UL_RATE_OFFLOAD_1_TIME && currDlRate > MIN_DL_RATE_OFFLOAD_1_TIME) {
                Log.i(TAG, "Decision 1: No previous remote executions. Good connectivity.");
                return true;
            } else {
                Log.i(TAG, "Decision 1: No previous remote executions. Bad connectivity.");
                return false;
            }
        }

        // Local part
        // Calculate the meanDurLocal and meanEnergyLocal from the previous runs.
        // Give more weight to recent measurements.
        Log.i(TAG, "------------ The local executions of the method:");
        int i = 0;
        Iterator<DBEntry> it = localResults.descendingIterator();
        while (it.hasNext()) {
//        for (DBEntry e : localResults) {
            DBEntry e = it.next();
            meanDurLocal += e.getExecDuration();
            meanEnergyLocal += e.getExecEnergy();
            Log.i(TAG, "------------ LocalEnergy: " + e.getExecEnergy());

            i++;
            if (i > 1) {
                meanDurLocal /= 2;
                meanEnergyLocal /= 2;
            }

            if (VERBOSE_LOG) {
                Log.i(TAG, "duration: " + meanDurLocal + " energy: " + meanEnergyLocal + " timestamp: "
                        + e.getTimestamp());
            }
        }
        Log.i(TAG, "nrLocalExec: " + nrLocalExec);
        Log.i(TAG, "meanDurLocal: " + meanDurLocal + "ns (" + meanDurLocal / 1000000000.0 + "s)");
        Log.i(TAG, "meanEnergyLocal: " + meanEnergyLocal);

        // Remote part
        long[] remoteDurations = new long[nrRemoteExec];
        long[] remoteEnergies = new long[nrRemoteExec];
        int[] remoteUlRates = new int[nrRemoteExec];
        int[] remoteDlRates = new int[nrRemoteExec];
        long[] remoteTimestamps = new long[nrRemoteExec];
        i = 0;
        if (VERBOSE_LOG) {
            Log.i(TAG, "------------ The remote executions of the method:");
        }
        Iterator<DBEntry> itRemote = remoteResults.descendingIterator();
//        for (DBEntry e : remoteResults) {
        while (itRemote.hasNext()) {
            DBEntry e = itRemote.next();
            remoteDurations[i] = e.getExecDuration();
            remoteEnergies[i] = e.getExecEnergy();
            remoteUlRates[i] = e.getUlRate();
            remoteDlRates[i] = e.getDlRate();
            remoteTimestamps[i] = e.getTimestamp();

            if (VERBOSE_LOG) {
                Log.i(TAG,
                        "duration: " + remoteDurations[i] + " energy: " + remoteEnergies[i] + " ulRate: "
                                + remoteUlRates[i] + " dlRate: " + remoteDlRates[i] + " timestamp: "
                                + remoteTimestamps[i]);
            }

            i++;
        }
        Log.i(TAG, "nrRemoteExec: " + nrRemoteExec);

        // DECISION 2
        int NR_TIMES_SWITCH_SIDES = 10;
        int count = 0;
        ExecLocation prevExecLocation = null;
        for (DBEntry e : dbCache.getAllEntriesFilteredOn(methodName)) {
            if (count < NR_TIMES_SWITCH_SIDES
                    && (prevExecLocation == null || e.getExecLocation().equals(prevExecLocation))) {
                prevExecLocation = e.getExecLocation();
                count++;
            } else {
                break;
            }
        }

        if (count == NR_TIMES_SWITCH_SIDES) {
            if (prevExecLocation.equals(ExecLocation.REMOTE)) {
                Log.i(TAG, "Decision 2: Too many remote executions in a row.");
                return false;
            } else if (prevExecLocation.equals(ExecLocation.LOCAL)) {
                Log.i(TAG, "Decision 2: Too many local executions in a row.");
                if (currUlRate > MIN_UL_RATE_OFFLOAD_1_TIME && currDlRate > MIN_DL_RATE_OFFLOAD_1_TIME) {
                    Log.i(TAG, "Decision 2->1: No previous remote executions. Good connectivity.");
                    return true;
                } else {
                    Log.i(TAG, "Decision 2->1: No previous remote executions. Bad connectivity.");
                    return false;
                }
            } else {
                Log.e(TAG, "Decision 2: This shouldn't happen, check the implementation.");
            }
        }

        // DECISION 3
        // Calculate two different mean values for the offloaded execution:
        // 1. The first are the same as for the local execution, gives more weight to recent runs
        // 2. The second are calculated as the average of the three closest values to the currentUlRate
        // and currDlRate
        int minDistIndex1 = 0, minDistIndex2 = 0, minDistIndex3 = 0;
        double minDist1 = Double.POSITIVE_INFINITY, minDist2 = Double.POSITIVE_INFINITY,
                minDist3 = Double.POSITIVE_INFINITY;
        for (i = 0; i < nrRemoteExec; i++) {
            // Calculate the first meanDuration and meanEnergy
            // The first formula is the same as for the local executions,
            // gives more importance to the last measurements.
            meanDurRemote1 += remoteDurations[i];
            meanEnergyRemote1 += remoteEnergies[i];
            if (i > 0) {
                meanDurRemote1 /= 2;
                meanEnergyRemote1 /= 2;
            }

            // Keep the indexes of the three measurements that have
            // the smallest distance dist(ulRate, dlRate, currUlRate, currDlRate)
            // minDist1 < minDist2 < minDist3
            double newDist = dist(remoteUlRates[i], remoteDlRates[i], currUlRate, currDlRate);
            if (newDist < minDist1) {
                minDist3 = minDist2;
                minDistIndex3 = minDistIndex2;

                minDist2 = minDist1;
                minDistIndex2 = minDistIndex1;

                minDist1 = newDist;
                minDistIndex1 = i;
            } else if (newDist < minDist2) {
                minDist3 = minDist2;
                minDistIndex3 = minDistIndex2;

                minDist2 = newDist;
                minDistIndex2 = i;
            } else if (newDist < minDist3) {
                minDist3 = newDist;
                minDistIndex3 = i;
            }
        }

        // Give more weight to the closest point
        meanDurRemote2 = (((remoteDurations[minDistIndex3] + remoteDurations[minDistIndex2]) / 2)
                + remoteDurations[minDistIndex1]) / 2;
        meanEnergyRemote2 = (((remoteEnergies[minDistIndex3] + remoteEnergies[minDistIndex2]) / 2)
                + remoteEnergies[minDistIndex1]) / 2;

        meanDurRemote = (meanDurRemote1 + meanDurRemote2) / 2;
        meanEnergyRemote = (meanEnergyRemote1 + meanEnergyRemote2) / 2;

        // Log.d(TAG, "meanDurRemote1: " + meanDurRemote1 + " meanDurRemote2: " + meanDurRemote2);
        Log.d(TAG, "meanDurRemote: " + meanDurRemote + "ns (" + meanDurRemote / 1000000000.0 + "s)");
        // Log.d(TAG,
        // "meanEnergyRemote1: " + meanEnergyRemote1 + " meanEnergyRemote2: " + meanEnergyRemote2);
        Log.d(TAG, "  meanEnergyRemote: " + meanEnergyRemote);

        Log.i(TAG, "Decision 3.");
        Log.d(TAG, "Making a choice for low energy and fast execution");
        return (meanDurRemote <= meanDurLocal) && (meanEnergyRemote <= meanEnergyLocal);
    }

    private double dist(int ul1, int dl1, int ul2, int dl2) {
        return Math.sqrt((ul2 - ul1) * (ul2 - ul1) + (dl2 - dl1) * (dl2 - dl1));
    }

    void setUserChoice(ExecLocation userChoice) {
        this.userChoice = userChoice;
    }

    public ExecLocation getUserChoice() {
        return this.userChoice;
    }

    public ExecLocation getLastExecLocation(String appName, String methodName) {
        Deque<DBEntry> results = dbCache.getAllEntriesFilteredOn(appName, methodName);
        return results.getFirst().getExecLocation();
    }

    public long getLastExecDuration(String appName, String methodName) {
        Deque<DBEntry> results = dbCache.getAllEntriesFilteredOn(appName, methodName);
        return results.getFirst().getExecDuration();
    }

    @SuppressWarnings("unused")
    private void sleep(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
