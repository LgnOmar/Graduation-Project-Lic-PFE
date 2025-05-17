import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'JibJobApp.dart';
import 'package:jibjob/screens/ClientSignUpPage.dart';


Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Supabase.initialize(
      url: 'https://pyeonzdwjabqzlujisdo.supabase.co',
      anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5ZW9uemR3amFicXpsdWppc2RvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU1OTEyNjgsImV4cCI6MjA2MTE2NzI2OH0.PT00SAxNBlzA4LRHoMyYFtHBytmCEm4CQPeWUTtUuqs',
  );

  runApp(
    MaterialApp( // Wrap your first page with MaterialApp
      title: 'JibJob', // Optional: Sets the title in the app switcher
      home: ClientSignUpPage(), // ClientSignUpPage is now a child of MaterialApp
      debugShowCheckedModeBanner: false, // Optional: Removes the debug banner
    ),
  );
}