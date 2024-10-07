package com.imashnake.aigo.ui.features

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionState
import com.google.accompanist.permissions.shouldShowRationale
import com.imashnake.aigo.R

private const val RATIONALE = "The app needs to see the Go board to digitize it. Please grant camera permission."
private const val REQUEST = "Camera permission is required for this feature."

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraPermissionDialog(
    permissionState: PermissionState,
    onDismiss: () -> Unit,
    launchRequest: () -> Unit,
    modifier: Modifier = Modifier,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            if (permissionState.status.shouldShowRationale) {
                TextButton(onClick = { onDismiss(); launchRequest() }) {
                    Text("Grant")
                }
            }
        },
        modifier = modifier,
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        },
        icon = {
            Icon(
                painter = painterResource(R.drawable.photo_camera),
                contentDescription = "Camera Permission",
            )
        },
        title = { Text("Camera Permission") },
        text = {
            Text(
                if (permissionState.status.shouldShowRationale) {
                    RATIONALE
                } else REQUEST
            )
        }
    )
}
