package dev.b5team.hack2progress.a2017.savelight;

import android.os.Build;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.annotation.RequiresApi;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class HomeFragment extends Fragment {
    public static final String TAG = HomeFragment.class.getSimpleName();
    private static final String BASE_URL = "http://efi-home-sergiowalls.c9users.io:8080";
    private View rootview;
    private WebView webView;

    public static HomeFragment newInstance() {
        return new HomeFragment();
    }

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        rootview = inflater.inflate(R.layout.home_fragment, container, false);

        setUpElements();
        setUpListeners();

        webView.getSettings().setJavaScriptEnabled(true);
        webView.loadUrl(BASE_URL);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                webView.setVisibility(View.VISIBLE);
            }

            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                Log.e(TAG, error.getErrorCode() + " _ " + error.getDescription());
            }
        });

        return rootview;
    }

    private void setUpElements() {
        webView = rootview.findViewById(R.id.webview_home_fragment);
    }

    private void setUpListeners() {

    }
}
