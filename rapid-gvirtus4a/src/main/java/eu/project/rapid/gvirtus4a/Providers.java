package eu.project.rapid.gvirtus4a;

import java.util.Vector;

/**
 * Created by raffaelemontella on 26/04/2017.
 */

public class Providers {
    private Vector<Provider> providers;
    private static Providers instance;

    private Providers() {
        providers = new Vector<>();
        providers.add(new Provider("193.205.230.23", 9991));
        instance = this;
    }

    public static Providers getInstance() {
        if (instance == null) {
            instance = new Providers();
        }
        return instance;
    }

    public Provider getBest() {
        return providers.get(0);
    }

    public void register(String host, int port) {
        Provider provider = new Provider(host, port);
        providers.add(provider);
    }
}
