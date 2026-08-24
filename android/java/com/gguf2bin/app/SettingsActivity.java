package com.gguf2bin.app;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.Toast;

public class SettingsActivity extends Activity {
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_settings);
        EditText ctx    = findViewById(R.id.ctx);
        EditText maxtok = findViewById(R.id.maxtok);
        EditText temp   = findViewById(R.id.temp);
        EditText topk   = findViewById(R.id.topk);

        SharedPreferences sp = getSharedPreferences("settings", MODE_PRIVATE);
        ctx.setText(String.valueOf(sp.getInt("ctx", 2048)));
        maxtok.setText(String.valueOf(sp.getInt("maxtok", 256)));
        temp.setText(String.valueOf(sp.getFloat("temp", 0.7f)));
        topk.setText(String.valueOf(sp.getInt("topk", 40)));

        findViewById(R.id.save).setOnClickListener(v -> {
            try {
                sp.edit()
                  .putInt("ctx", clamp(Integer.parseInt(ctx.getText().toString()), 512, 32768))
                  .putInt("maxtok", clamp(Integer.parseInt(maxtok.getText().toString()), 16, 4096))
                  .putFloat("temp", (float) Double.parseDouble(temp.getText().toString()))
                  .putInt("topk", clamp(Integer.parseInt(topk.getText().toString()), 1, 200))
                  .apply();
                Toast.makeText(this, "Guardado", 0).show();
                finish();
            } catch (Exception e) {
                Toast.makeText(this, "Valores inválidos", 0).show();
            }
        });
    }
    private static int clamp(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
}
