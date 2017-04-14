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
package eu.project.rapid.temp;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import eu.project.rapid.ac.DFE;

public class MainActivity extends Activity {

  private static final String TAG = "MainActivity";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    DFE dfe = new DFE();
    TestRemoteable t = new TestRemoteable(dfe);
    long result = t.factorial(10);

    Log.i(TAG, "The result is: " + result);

  }
}
