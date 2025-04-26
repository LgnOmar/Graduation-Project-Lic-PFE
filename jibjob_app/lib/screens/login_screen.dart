// Import the core Flutter Material library - provides widgets like Scaffold, Text, etc.
import 'package:flutter/material.dart';

// Define LoginScreen as a StatelessWidget (it won't change its own state initially)
class LoginScreen extends StatelessWidget {
  // 'const' constructor for performance optimization if the widget is truly constant
  const LoginScreen({super.key}); // 'key' helps Flutter identify widgets

  // The 'build' method describes how the widget should look
  @override
  Widget build(BuildContext context) {
    // Scaffold provides basic app structure (app bar, body)
    return Scaffold(
      // AppBar at the top (optional, maybe your design doesn't have one)
      appBar: AppBar(
        title: const Text('JibJob Login'), // Text displayed in the AppBar
        backgroundColor: Colors.blueAccent, // Example color
      ),
      // 'body' is the main content area of the screen
      body: Center( // Center the content vertically and horizontally
        child: Text(
          'Login Screen Content Here', // Placeholder text
          style: TextStyle(fontSize: 24), // Basic text styling
        ),
      ),
    );
  }
}
