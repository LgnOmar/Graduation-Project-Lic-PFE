class Offre {
  String? demande;
  String? phone;
  String? address;
  List<String?> images = [];
  static int nb = 0;

  Offre({
    this.demande,
    this.phone,
    this.address,
    List<String?>? images ,
  }) {
    this.images = images ?? [];
    nb++;
  }

  
}

List<Offre> Liste_Offres = [];