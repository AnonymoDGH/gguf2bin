package com.gguf2bin.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;

public class MainActivity extends Activity {
    private EditText urlBox;
    private Button downloadBtn;
    private TextView status;
    private ProgressBar progress;
    private ListView list;
    private ArrayAdapter<String> adapter;
    private final ArrayList<String> names = new ArrayList<>();

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_main);
        urlBox = findViewById(R.id.url);
        downloadBtn = findViewById(R.id.download);
        status = findViewById(R.id.status);
        progress = findViewById(R.id.dl_progress);
        list = findViewById(R.id.list);

        findViewById(R.id.settings).setOnClickListener(v ->
            startActivity(new Intent(this, SettingsActivity.class)));

        refresh();
        adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, names);
        list.setAdapter(adapter);
        list.setOnItemClickListener((p, v, pos, id) -> {
            Intent it = new Intent(this, ChatActivity.class);
            it.putExtra("model", modelsDir(names.get(pos)).getAbsolutePath());
            startActivity(it);
        });
        list.setOnItemLongClickListener((p, v, pos, id) -> {
            File f = modelsDir(names.get(pos));
            new android.app.AlertDialog.Builder(this)
                .setTitle(f.getName())
                .setMessage(String.format("Eliminar? (%.0f MB)", f.length() / 1048576.0))
                .setPositiveButton("Eliminar", (d, w) -> { f.delete(); refresh(); })
                .setNegativeButton("Cancelar", null).show();
            return true;
        });
        downloadBtn.setOnClickListener(v -> download());
    }

    @Override protected void onResume() { super.onResume(); refresh(); }

    private File modelsDir(String name) { return new File(getFilesDir(), name); }

    private void refresh() {
        names.clear();
        File[] fs = getFilesDir().listFiles();
        if (fs != null) for (File f : fs)
            if (f.getName().endsWith(".g2bx"))
                names.add(String.format("%s\n%.0f MB", f.getName(), f.length() / 1048576.0));
        if (adapter != null) adapter.notifyDataSetChanged();
    }

    private void download() {
        String u = urlBox.getText().toString().trim();
        if (!u.startsWith("http")) { Toast.makeText(this, "URL invalida", 1).show(); return; }
        String fname = u.substring(u.lastIndexOf('/') + 1);
        if (fname.isEmpty()) fname = "modelo.g2bx";
        if (!fname.endsWith(".g2bx")) fname += ".g2bx";
        File dst = modelsDir(fname);
        downloadBtn.setEnabled(false);
        progress.setIndeterminate(true);
        progress.setVisibility(View.VISIBLE);
        status.setText("Conectando...");
        new Thread(() -> {
            try {
                HttpURLConnection c = (HttpURLConnection) new URL(u).openConnection();
                c.setInstanceFollowRedirects(true);
                c.connect();
                long total = c.getContentLength();
                InputStream in = c.getInputStream();
                FileOutputStream out = new FileOutputStream(dst);
                byte[] buf = new byte[1 << 16];
                final long[] got = {0}; int r;
                runOnUiThread(() -> { progress.setIndeterminate(total <= 0); });
                while ((r = in.read(buf)) > 0) {
                    out.write(buf, 0, r); got[0] += r;
                    final int pct = total > 0 ? (int)(got[0] * 100 / total) : 0;
                    runOnUiThread(() -> {
                        if (total > 0) {
                            progress.setProgress(pct);
                            status.setText(String.format("%d%% · %.1f/%.1f MB", pct,
                                got[0] / 1048576.0, total / 1048576.0));
                        } else
                            status.setText(String.format("%.1f MB", got[0] / 1048576.0));
                    });
                }
                out.close(); in.close();
                runOnUiThread(() -> {
                    status.setText("Descargado"); progress.setVisibility(View.GONE);
                    downloadBtn.setEnabled(true); refresh();
                });
            } catch (Exception e) {
                dst.delete();
                runOnUiThread(() -> {
                    status.setText("Error: " + e.getMessage());
                    progress.setVisibility(View.GONE); downloadBtn.setEnabled(true);
                });
            }
        }).start();
    }
}
