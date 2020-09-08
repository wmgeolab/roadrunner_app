package com.example.roadrunner_geolab.myAlerts;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.support.v4.app.DialogFragment;
import com.example.roadrunner_geolab.R;

public class AlertDialogGPS extends DialogFragment {

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {

        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());

        builder.setMessage(R.string.dialog_gps_setting)
          .setPositiveButton(R.string.settings, new DialogInterface.OnClickListener() {
              @Override
              public void onClick(DialogInterface dialog, int which) {
                  Intent onGPS = new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
                  startActivity(onGPS);
              }
          })
          .setNegativeButton(R.string.cancel, new DialogInterface.OnClickListener() {
              @Override
              public void onClick(DialogInterface dialog, int which) {
                  // User cancelled.
              }
          });
        return builder.create();
    }
}
