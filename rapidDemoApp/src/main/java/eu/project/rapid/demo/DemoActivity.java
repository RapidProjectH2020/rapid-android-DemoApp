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
package eu.project.rapid.demo;

import android.app.Activity;
import android.app.ProgressDialog;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Locale;
import java.util.Random;

import eu.project.rapid.ac.DFE;
import eu.project.rapid.common.Clone;
import eu.project.rapid.common.RapidConstants;
import eu.project.rapid.common.RapidConstants.COMM_TYPE;
import eu.project.rapid.gvirtus.MatrixMul;
import eu.project.rapid.gvirtus4a.Provider;
import eu.project.rapid.gvirtus4a.Providers;
import eu.project.rapid.queens.NQueens;
import eu.project.rapid.sudoku.Sudoku;
import eu.project.rapid.synthBenchmark.JniTest;

/**
 * The class that handles configuration parameters and starts the offloading process.
 *
 * @author sokol
 */
public class DemoActivity extends Activity implements DFE.DfeCallback {

    private static final String TAG = "DemoActivity";

    private static int nrVMs = 1;
    private TextView textViewVmConnected;
    private Handler handler;

    private DFE dfe;

    private int nQueensLocalNr;
    private double nQueensLocalTotDur;
    private int nQueensRemoteNr;
    private double nQueensRemoteTotDur;
    private TextView nQueensLocalNrText;
    private TextView nQueensLocalDurText;
    private TextView nQueensRemoteNrText;
    private TextView nQueensRemoteDurText;

    private int jniLocalNr;
    private double jniLocalTotDur;
    private int jniRemoteNr;
    private double jniRemoteTotDur;
    private TextView jniLocalNrText;
    private TextView jniLocalDurText;
    private TextView jniRemoteNrText;
    private TextView jniRemoteDurText;

    private int gvirtusLocalNr;
    private double gvirtusLocalTotDur;
    private int gvirtusRemoteNr;
    private double gvirtusRemoteTotDur;
    private TextView gvirtusLocalNrText;
    private TextView gvirtusLocalDurText;
    private TextView gvirtusRemoteNrText;
    private TextView gvirtusRemoteDurText;

    private CheckBox checkBoxEnforceForwarding;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        Log.i(TAG, "onCreate");

        String vmIp = getIntent().getStringExtra(MainActivity.KEY_VM_IP);
        boolean useRapidInfrastructure = getIntent().getBooleanExtra(MainActivity.KEY_USE_RAPID_INFRASTRUCTURE, true);
        COMM_TYPE commType = (COMM_TYPE) getIntent().getSerializableExtra(MainActivity.KEY_CONN_TYPE);
        boolean usePrevVm = getIntent().getBooleanExtra(MainActivity.KEY_USE_PREV_VM, true);

        // The handler for
        handler = new Handler();

        // Used for specifying the number of VMs the developer wants to use.
        LinearLayout layoutNrVMs = (LinearLayout) findViewById(R.id.layoutNrVMs);
        if (useRapidInfrastructure) {
            layoutNrVMs.setVisibility(View.VISIBLE);
            Spinner nrVMsSpinner = (Spinner) findViewById(R.id.spinnerNrVMs);
            nrVMsSpinner.setOnItemSelectedListener(new NrVMsSelectedListener());
        } else {
            layoutNrVMs.setVisibility(View.GONE);
        }

        textViewVmConnected = (TextView) findViewById(R.id.textVmConnectionStatus);

        nQueensLocalNrText = (TextView) findViewById(R.id.valNQueensLocalNr);
        nQueensLocalDurText = (TextView) findViewById(R.id.valNQueensLocalTime);
        nQueensRemoteNrText = (TextView) findViewById(R.id.valNQueensRemoteNr);
        nQueensRemoteDurText = (TextView) findViewById(R.id.valNQueensRemoteTime);

        jniLocalNrText = (TextView) findViewById(R.id.valJNILocalNr);
        jniLocalDurText = (TextView) findViewById(R.id.valJNILocalTime);
        jniRemoteNrText = (TextView) findViewById(R.id.valJNIRemoteNr);
        jniRemoteDurText = (TextView) findViewById(R.id.valJNIRemoteTime);

        gvirtusLocalNrText = (TextView) findViewById(R.id.valGvirtusLocalNr);
        gvirtusLocalDurText = (TextView) findViewById(R.id.valGvirtusLocalTime);
        gvirtusRemoteNrText = (TextView) findViewById(R.id.valGvirtusRemoteNr);
        gvirtusRemoteDurText = (TextView) findViewById(R.id.valGvirtusRemoteTime);

        checkBoxEnforceForwarding = (CheckBox) findViewById(R.id.checkboxEnforceForwarding);

        // If we don't specify the IP of the VM, we assume that we are using the Rapid infrastructure,
        // i.e. the DS, the VMM, the SLAM, etc., which means that the DFE will select automatically a
        // VM. We let the user select a VM manually for fast deploy and testing.
        if (vmIp == null) {
            dfe = DFE.getInstance(getPackageName(), getPackageManager(), this);
        } else {
            dfe = DFE.getInstance(getPackageName(), getPackageManager(), this,
                    new Clone("", vmIp), commType);
        }
    }

    private class VmConnectionStatusUpdater implements Runnable {

        private COMM_TYPE commType;
        private boolean isConnected;

        VmConnectionStatusUpdater(boolean isConnected, COMM_TYPE commType) {
            this.isConnected = isConnected;
            this.commType = commType;
        }

        public void run() {
            handler.post(new Runnable() {
                public void run() {
                    textViewVmConnected.setTextColor(Color.GREEN);
                    if (isConnected) {
                        if (commType.equals(COMM_TYPE.CLEAR)) {
                            textViewVmConnected.setText(R.string.textVmConnectedClear);
                        } else if (commType.equals(COMM_TYPE.SSL)) {
                            textViewVmConnected.setText(R.string.textVmConnectedSSL);
                        }
                    } else {
                        textViewVmConnected.setTextColor(Color.RED);
                        textViewVmConnected.setText(R.string.textVmDisconnected);
                    }
                }
            });
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.i(TAG, "onDestroy");

        if (dfe != null) {
            dfe.onDestroy();
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        Log.d(TAG, "OnPause");
    }

    public void onClickJni1(View v) {
        JniTest jni = new JniTest(dfe);

        String result = jni.jniCaller();
        Log.i(TAG, "Result of jni invocation: " + result);

        Toast.makeText(DemoActivity.this, result, Toast.LENGTH_SHORT).show();

        String methodName = "localjniCaller";
        if (dfe.getLastExecLocation(getPackageName(), methodName).equals(RapidConstants.ExecLocation.LOCAL)) {
            jniLocalNrText.setText(String.format(Locale.ENGLISH, "%d", ++jniLocalNr));
            jniLocalTotDur += dfe.getLastExecDuration(getPackageName(), methodName);
            jniLocalDurText.setText(String.format(Locale.ENGLISH, "%.2f", jniLocalTotDur / jniLocalNr / 1000000));
        } else {
            jniRemoteNrText.setText(String.format(Locale.ENGLISH, "%d", ++jniRemoteNr));
            jniRemoteTotDur += dfe.getLastExecDuration(getPackageName(), methodName);
            jniRemoteDurText.setText(String.format(Locale.ENGLISH, "%.2f", jniRemoteTotDur / jniRemoteNr / 1000000));
        }
    }

    public void onClickSudoku(View v) {

        Sudoku sudoku = new Sudoku(dfe);

        boolean result = sudoku.hasSolution();

        if (result) {
            Toast.makeText(DemoActivity.this, "Sudoku has solution", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(DemoActivity.this, "Sudoku does not have solution", Toast.LENGTH_SHORT)
                    .show();
        }
    }

    public void onClickQueenSolver(View v) {
            new NQueensTask().execute();
    }

    private class NQueensTask extends AsyncTask<Void, Void, Integer> {
        int nrQueens;
        boolean isEnforceForwarding = checkBoxEnforceForwarding.isChecked();
        Spinner nrQueensSpinner = (Spinner) findViewById(R.id.spinnerNrQueens);
        // Show a spinning dialog while solving the puzzle
        ProgressDialog pd = ProgressDialog.show(DemoActivity.this, "Working...", "Solving N Queens...", true, false);

        NQueensTask() {
            this.nrQueens = Integer.parseInt((String) nrQueensSpinner.getSelectedItem());
        }

        public NQueensTask(int nrQueens) {
            this.nrQueens = nrQueens;
            // Show a spinning dialog while solving the puzzle
            pd = ProgressDialog.show(DemoActivity.this, "Working...", "Solving N Queens...", true, false);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            NQueens puzzle = new NQueens(dfe, nrVMs);
            if (checkBoxEnforceForwarding != null) {
                puzzle.setEnforceForwarding(isEnforceForwarding);
            }

            return puzzle.solveNQueens(nrQueens);
        }

        @Override
        protected void onPostExecute(Integer result) {
            Log.i(TAG, "Finished execution");
            if (pd != null) {
                pd.dismiss();
            }

            String methodName = "localSolveNQueens";
            Log.i(TAG, nrQueens + "-Queens solved, solutions: " + result);

            Toast.makeText(DemoActivity.this, nrQueens + "-Queens solved, solutions: " + result,
                    Toast.LENGTH_SHORT).show();

            if (dfe.getLastExecLocation(getPackageName(), methodName).equals(RapidConstants.ExecLocation.LOCAL)) {
                nQueensLocalNrText.setText(String.format(Locale.ENGLISH, "%d", ++nQueensLocalNr));
                nQueensLocalTotDur += dfe.getLastExecDuration(getPackageName(), methodName);
                nQueensLocalDurText.setText(String.format(Locale.ENGLISH, "%.2f", nQueensLocalTotDur / nQueensLocalNr / 1000000));
            } else {
                nQueensRemoteNrText.setText(String.format(Locale.ENGLISH, "%d", ++nQueensRemoteNr));
                nQueensRemoteTotDur += dfe.getLastExecDuration(getPackageName(), methodName);
                nQueensRemoteDurText.setText(String.format(Locale.ENGLISH, "%.2f", nQueensRemoteTotDur / nQueensRemoteNr / 1000000));
            }

            if (checkBoxEnforceForwarding != null) {
                checkBoxEnforceForwarding.setChecked(false);
            }
        }
    }

    public void onClickMultipleQueenSolver(View v) {
        new NQueensMultipleTask().execute();
    }

    private class NQueensMultipleTask extends AsyncTask<Void, Void, Void> {
        // Show a spinning dialog while solving the puzzle
        ProgressDialog pd = ProgressDialog.show(DemoActivity.this, "Working...", "Solving multiple N Queens...", true, false);

        @Override
        protected Void doInBackground(Void... params) {
            final Random r = new Random();
            for (int i = 0; i < 10; i++) {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        NQueens puzzle = new NQueens(dfe, nrVMs);
                        int nrQueens = 4 + r.nextInt(4);
                        Log.v(Thread.currentThread().getName(), "Started " + nrQueens + "-queens");
                        int result = puzzle.solveNQueens(nrQueens);
                        Log.v(Thread.currentThread().getName(), "Finished " + nrQueens + "-queens, " + result + " solutions");
                    }
                }).start();
            }

            return null;
        }

        @Override
        protected void onPostExecute(Void param) {
            Log.i(TAG, "Finished execution");
            if (pd != null) {
                pd.dismiss();
            }
        }
    }

    public void onClickGvirtusDemo(View v) {
        new GvirtusCaller().execute();
    }

    private class GvirtusCaller extends AsyncTask<Void, Void, Void> {
        // Show a spinning dialog while running the GVirtuS demo
        ProgressDialog pd = ProgressDialog.show(DemoActivity.this, "Working...",
                "Running the GVirtuS demo...", true, false);

        @Override
        protected Void doInBackground(Void... params) {
            int nrTests = 1;



            MatrixMul matrixMul = new MatrixMul(dfe);
            int wa = 8;
            int wb = 12;

            for (int i = 0; i < nrTests; i++) {
                Log.i(TAG, "------------ Started running GVirtuS with DFE.");
                boolean result = matrixMul.gpuMatrixMul(wa, wb, wa);
                Log.i(TAG, "Finished GVirtuS matrixMul with DFE: isResultValid=" + result);
            }

            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            Log.i(TAG, "Finished execution");
            if (pd != null) {
                pd.dismiss();
            }

            String methodName = "localGpuMatrixMul";
            if (dfe.getLastExecLocation(getPackageName(), methodName).equals(RapidConstants.ExecLocation.LOCAL)) {
                gvirtusLocalNrText.setText(String.format(Locale.ENGLISH, "%d", ++gvirtusLocalNr));
                gvirtusLocalTotDur += dfe.getLastExecDuration(getPackageName(), methodName);
                gvirtusLocalDurText.setText(String.format(Locale.ENGLISH, "%.2f", gvirtusLocalTotDur / gvirtusLocalNr / 1000000));
            } else {
                gvirtusRemoteNrText.setText(String.format(Locale.ENGLISH, "%d", ++gvirtusRemoteNr));
                gvirtusRemoteTotDur += dfe.getLastExecDuration(getPackageName(), methodName);
                gvirtusRemoteDurText.setText(String.format(Locale.ENGLISH, "%.2f", gvirtusRemoteTotDur / gvirtusRemoteNr / 1000000));
            }
        }
    }

    public void onRadioExecLocationChecked(View radioButton) {
        switch (radioButton.getId()) {

            case R.id.radio_local:
                dfe.setUserChoice(RapidConstants.ExecLocation.LOCAL);
                break;

            case R.id.radio_remote:
                dfe.setUserChoice(RapidConstants.ExecLocation.REMOTE);
                break;

            case R.id.radio_exec_time_energy:
                dfe.setUserChoice(RapidConstants.ExecLocation.DYNAMIC);
                break;
        }
    }

    private class NrVMsSelectedListener implements OnItemSelectedListener {

        public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {

            nrVMs = Integer.parseInt((String) parent.getItemAtPosition(pos));
            Log.i(TAG, "Number of VMs: " + nrVMs);
            dfe.setNrVMs(nrVMs);
        }

        public void onNothingSelected(AdapterView<?> arg0) {
            Log.i(TAG, "Nothing selected on VM spinner");
        }
    }

    private void sleep(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public void vmConnectionStatusUpdate(boolean isConnected, COMM_TYPE commType) {
        new Thread(new VmConnectionStatusUpdater(isConnected, commType)).start();
    }

    public void onCheckboxClicked(View view) {
        // Is the view now checked?
        boolean checked = ((CheckBox) view).isChecked();

        // Check which checkbox was clicked
        switch(view.getId()) {
            case R.id.checkboxEnforceForwarding:
                if (checked) {
                    Log.i(TAG, "NQueens will enforce forwarding if run on the VM");
                } else {
                    Log.i(TAG, "NQueens will NOT enforce forwarding");
                }
                break;
        }
    }

    private void registerProviders() {
        Providers providers=Providers.getInstance();
        providers.unregister();
        providers.register("80.158.23.133", 9998);
        providers.register("54.72.110.23", 9998);
        providers.register("193.205.230.23", 9998);
        providers.setDefaultProvider(providers.getBest(0650));
    }
}
