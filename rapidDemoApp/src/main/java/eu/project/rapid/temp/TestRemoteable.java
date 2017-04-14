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

import java.lang.reflect.Method;

import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.Remote;
import eu.project.rapid.ac.Remoteable;

public class TestRemoteable extends Remoteable {
  private static final long serialVersionUID = 2305945752914046079L;

  private DFE dfe;

  public TestRemoteable(DFE dfe) {
    this.dfe = dfe;
  }

  public long factorial(int n) {
    Class<?>[] paramTypes = {int.class};
    Object[] paramValues = {n};
    long result = -1;

    try {
      Method toExecute = this.getClass().getDeclaredMethod("originalFactorial", paramTypes);
      result = (Long) dfe.execute(toExecute, paramValues, this);
    } catch (SecurityException e) {
      e.printStackTrace();
      throw e;
    } catch (NoSuchMethodException e) {
      e.printStackTrace();
    } catch (Throwable e) {
      e.printStackTrace();
    }

    return result;
  }

  @Remote
  public long originalFactorial(int n) {
    long result = 1;
    for (int i = 2; i <= n; i++) {
      result *= 2;
    }
    return result;
  }

  @Override
  public void copyState(Remoteable state) {}
}


