import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'JibJobApp.dart';


Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Supabase.initialize(
      url: 'https://pyeonzdwjabqzlujisdo.supabase.co',
      anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5ZW9uemR3amFicXpsdWppc2RvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU1OTEyNjgsImV4cCI6MjA2MTE2NzI2OH0.PT00SAxNBlzA4LRHoMyYFtHBytmCEm4CQPeWUTtUuqs',
  );
  
  runApp(JibJobApp());
}