import 'package:flutter/material.dart';
import 'JibJobAuthPage.dart';
import 'Profilepro.dart' ;

class ProSignUpPage extends StatefulWidget {
  @override
  _ProSignUpPageState createState() => _ProSignUpPageState();
}

class _ProSignUpPageState extends State<ProSignUpPage> {

  final Color darkPurple = Color(0xFF20004E);

  final List<Map<String, dynamic>> categories = [
    {
      "name": "Maison",
      "services": [
        {"name": "Plombier", "image": "assets/plombier.png" },
        {"name" : "Peintre" , "image": "assets/peintre.png"},
        {"name" : "Électricien" , "image": "assets/electricien.png"}
      ]
    },
    {
      "name": "Auto",
      "services": [
        {"name" : "Mécanicien", "image": "assets/mecanicien.png"},
        {"name" : "Tolier", "image": "assets/tolier.png"},
        {"name" : "Lavage", "image": "assets/lavage.png"}
      ]
    },
    {
      "name": "Beauté",
      "services": [
        {"name" : "Coiffeur", "image": "assets/coiffeur.png"},
        {"name" : "Esthéticienne", "image": "assets/estheticienne.png"},
        {"name" : "Maquilleur", "image": "assets/maquilleur.png"}
      ]
    },
    {
      "name": "Informatique",
      "services": [
        {"name" : "Développeur", "image": "assets/developeur.png"},
        {"name" : "Réparateur", "image": "assets/reparation.png"},
        {"name" : "Technicien réseau", "image": "assets/technicien.png"}
      ]
    },
    {
      "name": "Éducation",
      "services": [
        {"name" : "Prof particulier", "image": "assets/plombier.png"},
        {"name" : "Coach scolaire", "image": "assets/plombier.png"},
        {"name" : "Formateur", "image": "assets/plombier.png"}
      ]
    },
    {
      "name": "Nettoyage",
      "services": [
        {"name" : "Ménage", "image": "assets/plombier.png"},
        {"name" : "Vitres", "image": "assets/plombier.png"},
        {"name" : "Tapis", "image": "assets/plombier.png"}
      ]
    },
  ];

  Set<String> selectedServices = {};


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16.0),
          child: ListView(
            children: [
              // Header
              Row(
                children: [
                  IconButton(
                    icon: Icon(Icons.arrow_back, color: darkPurple),
                    onPressed: () {
                      runApp(
                          MaterialApp(
                              home : JibJobAuthPage() ,
                              debugShowCheckedModeBanner: false
                          )) ;
                    },
                  ),
                  SizedBox(width: 10),
                  Text(
                    "Creer un compte pro",
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                      color: darkPurple,
                    ),
                  )
                ],
              ),
              SizedBox(height: 24),

              // Presentation
              Text("Presentation",
                  style: TextStyle(fontWeight: FontWeight.bold)),
              Text("Presenter vous en quelques mots aux clients"),
              SizedBox(height: 8),
              TextField(
                maxLines: 4,
                decoration: InputDecoration(
                  hintText:
                  "ex: j'ai x années d’expérience en x. Le peux fournir le service a...",
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
              SizedBox(height: 24),

              // Photo de profile
              Text("Photo de Profile",
                  style: TextStyle(fontWeight: FontWeight.bold)),
              Text("Photo de votre visage"),
              SizedBox(height: 12),
              CircleAvatar(
                radius: 40,
                backgroundColor: Colors.grey[300],
                child: Icon(Icons.camera_alt, size: 30, color: Colors.grey[700]),
              ),
              SizedBox(height: 24),

              // Realisations
              Text("Realisations",
                  style: TextStyle(fontWeight: FontWeight.bold)),
              Text("Montrez aux clients des photos de vos travaux"),
              SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: List.generate(
                  6,
                      (index) => Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Icon(Icons.camera_alt),
                  ),
                ),
              ),
              SizedBox(height: 24),

              // Services

              ...categories.map((category) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(height: 24),
                    Row(
                      children: [
                        Icon(Icons.category, color: darkPurple, size: 18),
                        SizedBox(width: 6),
                        Text(category["name"]),
                      ],
                    ),
                    SizedBox(height: 12),
                    Wrap(
                      spacing: 12,
                      runSpacing: 12,
                      children: category["services"].map<Widget>((service) {
                        String name = service["name"];
                        String imagePath = service["image"];
                        bool isSelected = selectedServices.contains(name);

                        return GestureDetector(
                          onTap: () {
                            setState(() {
                              if (isSelected) {
                                selectedServices.remove(name);
                              } else {
                                selectedServices.add(name);
                              }
                            });
                          },
                          child: Column(
                            children: [
                              CircleAvatar(
                                radius: 40,
                                backgroundColor: isSelected ? Colors.deepPurple : Colors.grey[300],
                                backgroundImage: AssetImage(imagePath),
                                child: isSelected
                                    ? Icon(Icons.check, color: Colors.white, size: 32)
                                    : null,
                              ),
                              SizedBox(height: 6),
                              Text(
                                name,
                                textAlign: TextAlign.center,
                                style: TextStyle(fontSize: 12),
                              ),
                            ],
                          ),
                        );
                      }).toList(),
                    ),
                  ],
                );
              }).toList(),
              SizedBox(height: 32),

              ElevatedButton(
                onPressed: () {runApp(
                    MaterialApp(
                        home : Profilepro() ,
                        debugShowCheckedModeBanner: false
                    )) ;
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: darkPurple,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  padding: EdgeInsets.symmetric(vertical: 14),
                ),
                child: Text("Terminer"),
              ),
              SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}