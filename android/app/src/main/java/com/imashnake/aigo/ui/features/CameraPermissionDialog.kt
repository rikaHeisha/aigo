package com.imashnake.aigo.ui.features

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionState
import com.google.accompanist.permissions.shouldShowRationale
import com.imashnake.aigo.ui.components.AigoDialog

private const val RATIONALE = "The app needs to see the Go board to digitize it\nplease grant camera permission"
private const val REQUEST = "Camera permission is required for this feature"

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraPermissionDialog(
    permissionState: PermissionState,
    onDismiss: () -> Unit,
    launchRequest: () -> Unit,
    modifier: Modifier = Modifier,
) {
    AigoDialog(onDismiss, modifier) {
        Text(
            if (permissionState.status.shouldShowRationale) {
                RATIONALE
            } else REQUEST
        )
        Row(horizontalArrangement = Arrangement.End) {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
            if (permissionState.status.shouldShowRationale) {
                TextButton(onClick = { onDismiss(); launchRequest() }) {
                    Text("Grant")
                }
            }
        }
    }
}