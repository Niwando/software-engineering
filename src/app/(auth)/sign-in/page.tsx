import { SignInForm } from "@/src/components/sign/sign-in-form";

export default function SignIn() {

  return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-6 bg-cover bg-center bg-no-repeat p-6 md:p-10">
        <img
          src="/dashboard.jpeg"
          alt="Dashboard Background"
          className="absolute top-0 left-0 -z-10 h-full w-full object-cover filter blur-sm"
        />
        <div className="flex w-full max-w-sm flex-col gap-6">
          <SignInForm />
        </div>
      </div>
  );
}
