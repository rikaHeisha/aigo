package com.imashnake.aigo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.imashnake.aigo.ui.features.CameraOverlay
import com.imashnake.aigo.ui.features.Home
import com.imashnake.aigo.ui.theme.AigoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            AigoTheme {
                MainScreen(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(MaterialTheme.colorScheme.background)
                )
            }
        }
    }
}

@Composable
fun MainScreen(modifier: Modifier = Modifier) {
    val navController = rememberNavController()

    Box(
        modifier = modifier,
        contentAlignment = Alignment.Center
    ) {
        NavHost(navController, startDestination = Home) {
            composable<Home> {
                Home(onCameraRequested = { navController.navigate(CameraOverlay) })
            }
            composable<CameraOverlay> {
                CameraOverlay()
            }
        }
    }
}
