package com.gguf2bin.app;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
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
    private ListView list;
    private ArrayAdapter<String> adapter;
    private ArrayList<String> names = new ArrayList<>();

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_main);
        urlBox = findViewById(R.id.url);
        downloadBtn = findViewById(R.id.download);
        status = findViewById(R.id.status);
        list = findViewById(R.id.list);
        refresh();
        adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, names);
        list.setAdapter(adapter);
        list.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            public void onItemClick(AdapterView<?> p, View v, int pos, long id) {
                File f = modelsDir(names.get(pos));
                android.content.Intent it = new android.content.Intent(MainActivity.this, ChatActivity.class);
                it.putExtra("model", f.getAbsolutePath());
                startActivity(it);
            }
        });
        downloadBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { download(); }
        });
    }

    private File modelsDir(String name) { return new File(getFilesDir(), name); }

    private void refresh() {
        names.clear();
        File[] fs = getFilesDir().listFiles();
        if (fs != null) for (File f : fs) if (f.getName().endsWith(".g2bx")) names.add(f.getName());
        if (adapter != null) adapter.notifyDataSetChanged();
    }

    private void download() {
        final String u = urlBox.getText().toString().trim();
        if (!u.startsWith("http")) { Toast.makeText(this, "URL inválida", 1).show(); return; }
        String fname = u.substring(u.lastIndexOf('/') + 1);
        if (!fname.endsWith(".g2bx")) fname += ".g2bx";
        final File dst = modelsDir(fname);
        downloadBtn.setEnabled(false);
        status.setText("Descargando…");
        new Thread(new Runnable() { public void run() {
            try {
                HttpURLConnection c = (HttpURLConnection) new URL(u).openConnection();
                c.setFollowRedirects(true);
                c.connect();
                long total = c.getContentLength();
                InputStream in = c.getInputStream();
                FileOutputStream out = new FileOutputStream(dst);
                byte[] buf = new byte[1 << 16];
                long got = 0; int r;
                while ((r = in.read(buf)) > 0) {
                    out.write(buf, 0, r); got += r;
                    final String s = total > 0
                        ? String.format("Descargando… %.1f / %.1f MB", got / 1048576.0, total / 1048576.0)
                        : String.format("Descargando… %.1f MB", got / 1048576.0);
                    runOnUiThread(new Runnable() { public void run() { status.setText(s); } });
                }
                out.close(); in.close();
                runOnUiThread(new Runnable() { public void run() {
                    status.setText("Listo"); downloadBtn.setEnabled(true); refresh();
                }});
            } catch (final Exception e) {
                dst.delete();
                runOnUiThread(new Runnable() { public void run() {
                    status.setText("Error: " + e.getMessage()); downloadBtn.setEnabled(true);
                }});
            }
        }}).start();
    }
}
