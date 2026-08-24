package com.gguf2bin.app;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

public class ChatActivity extends Activity {
    private long model = 0;
    private TextView out;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final StringBuilder history = new StringBuilder();

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_chat);
        out = findViewById(R.id.out);
        scroll = findViewById(R.id.scroll);
        input = findViewById(R.id.input);
        send = findViewById(R.id.send);
        final String path = getIntent().getStringExtra("model");
        setTitle(new java.io.File(path).getName());
        new Thread(new Runnable() { public void run() {
            final long ptr = Native.loadModel(path, 2048);
            runOnUiThread(new Runnable() { public void run() {
                if (ptr == 0) { finish(); return; }
                model = ptr;
                out.setText("Modelo cargado. Escribe tu mensaje.\n");
            }});
        }}).start();

        send.setOnClickListener(new View.OnClickListener() { public void onClick(View v) {
            final String q = input.getText().toString().trim();
            if (q.isEmpty() || model == 0) return;
            input.setText("");
            send.setEnabled(false);
            history.append("\nUsuario: ").append(q).append("\nAsistente: ");
            ui.post(new Runnable() { public void run() {
                out.append("\n\nTú: " + q + "\nBot: ");
                scroll.fullScroll(View.FOCUS_DOWN);
            }});
            new Thread(new Runnable() { public void run() {
                final String reply = Native.generate(model, history.toString(), 128, 0.7f,
                    new Native.TokenSink() { public void onToken(final String t) {
                        ui.post(new Runnable() { public void run() {
                            out.append(t);
                            scroll.fullScroll(View.FOCUS_DOWN);
                        }});
                    }});
                if (reply != null) history.append(reply);
                ui.post(new Runnable() { public void run() { send.setEnabled(true); } });
            }}).start();
        }});
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        if (model != 0) { Native.freeModel(model); model = 0; }
    }
}
